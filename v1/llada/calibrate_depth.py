"""Layer-redundancy calibration -- `01_experiment_plan.md` section 1, `profile_redundancy`.

Runs the sampler on calibration prompts with a forward hook on every block capturing
cos(h_in, h_out) on the *masked positions of the current block*, bucketed by mask ratio.
A layer whose output points where its input already pointed changed little, so a high cosine
means high redundancy. This is the ranking Goel et al. (2603.07475) use for static skipping
and the one `06` section 2 step 4 asks us to reproduce with.

Two things this deliberately does NOT do:

  * It does not average over all positions. The quantity that matters is what the layer does
    to the tokens the sampler is about to commit, and those are the masked ones inside the
    current block; averaging over the prompt would be dominated by clean context.
  * It does not pick the skip set. That is `depth_schedule.select_skips`, which applies the
    non-adjacency and first/last rules the ceilings of `06` section 2.5 Stage 0 are built on.
    Keeping ranking and selection apart is what lets the same ranking feed static-k (step 4)
    and the per-bucket iso-error budget (schedule B).

    python calibrate_depth.py --n-prompts 128 --n-buckets 4 --out <json>
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate as G                                            # noqa: E402
import profile_step as P                                        # noqa: E402
from model.modeling_llada import LLaDAModelLM                   # noqa: E402
from transformers import AutoTokenizer                          # noqa: E402

MASK_ID = 126336


class CosineProbe:
    """cos(h_in, h_out) per block, on the masked positions of the current block."""

    def __init__(self, blocks, n_buckets):
        self.blocks, self.n_buckets = blocks, n_buckets
        self.sum = [[0.0] * len(blocks) for _ in range(n_buckets)]
        self.cnt = [[0] * len(blocks) for _ in range(n_buckets)]
        self.pos = None          # bool mask over the pass's positions, or None to skip
        self.bucket = 0
        self._h = []

    def _pre(self, idx):
        def f(mod, args, kwargs=None):
            self._in = args[0] if args else kwargs["x"]
        return f

    def _post(self, idx):
        def f(mod, args, output, kwargs=None):
            if self.pos is None:
                return
            h_in, h_out = self._in, output[0]
            if self.pos.shape[-1] != h_in.shape[1]:
                return
            sel = self.pos.view(-1)
            a = h_in[0][sel].float()
            b = h_out[0][sel].float()
            if a.numel() == 0:
                return
            c = F.cosine_similarity(a, b, dim=-1).mean().item()
            self.sum[self.bucket][idx] += c
            self.cnt[self.bucket][idx] += 1
        return f

    def __enter__(self):
        for i, blk in enumerate(self.blocks):
            self._h.append(blk.register_forward_pre_hook(self._pre(i)))
            self._h.append(blk.register_forward_hook(self._post(i)))
        return self

    def __exit__(self, *a):
        for h in self._h:
            h.remove()
        self._h.clear()

    def matrix(self):
        return [[(self.sum[b][l] / self.cnt[b][l]) if self.cnt[b][l] else float("nan")
                 for l in range(len(self.blocks))] for b in range(self.n_buckets)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--n-buckets", type=int, default=4)
    ap.add_argument("--split", default="train", help="calibration set, disjoint from eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    blk_name, blocks = P.find_blocks(model)
    L = len(blocks)

    # calibration prompts come from GSM8K *train* (`06` section 1 item 4), never the eval set
    tr_q, tr_a = P._gsm8k("train")
    shots = "".join(P.FEWSHOT_TEMPLATE.format(q=tr_q[i], a=tr_a[i]) for i in range(a.n_shot))
    prompts, calib_ids = [], list(range(a.n_shot, a.n_shot + a.n_prompts))
    for i in calib_ids:
        user = shots + f"Question: {tr_q[i]}\nAnswer:"
        prompts.append(tok.apply_chat_template([{"role": "user", "content": user}],
                                               add_generation_prompt=True, tokenize=False))

    probe = CosineProbe(blocks, a.n_buckets)

    def run(text):
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        Lp = ids.shape[1]
        x = torch.full((1, Lp + a.gen_length), MASK_ID, dtype=torch.long, device=dev)
        x[:, :Lp] = ids
        for nb in range(a.gen_length // a.block_length):
            s = Lp + nb * a.block_length
            e = s + a.block_length
            for _ in range(a.block_length):
                masked = (x[:, s:e] == MASK_ID)
                if masked.sum() == 0:
                    break
                r = masked.float().mean().item()
                probe.bucket = min(int((1.0 - r) * a.n_buckets), a.n_buckets - 1)
                probe.pos = torch.zeros_like(x, dtype=torch.bool)
                probe.pos[:, s:e] = masked
                logits = model(x, use_cache=False).logits
                probe.pos = None
                mask_all = (x == MASK_ID); mask_all[:, e:] = False
                x0, ti = G.get_transfer_index(logits, 0.0, "low_confidence", mask_all, x,
                                              None, 0.9)
                x = torch.where(ti, x0, x)

    with torch.no_grad(), probe:
        for i, t in enumerate(prompts):
            run(t)
            if (i + 1) % 8 == 0:
                print(f"  {i+1}/{len(prompts)} prompts", flush=True)

    mat = probe.matrix()
    order = [sorted(range(L), key=lambda l, b=b: -mat[b][l]) for b in range(a.n_buckets)]
    global_mean = [sum(mat[b][l] for b in range(a.n_buckets)) / a.n_buckets for l in range(L)]
    res = dict(model=a.model, n_layers=L, block_module=blk_name, args=vars(a),
               gpu=torch.cuda.get_device_name(0),
               calib_ids=calib_ids, fewshot_ids=list(range(a.n_shot)),
               cosine=mat, skip_order=order,
               global_cosine=global_mean,
               global_order=sorted(range(L), key=lambda l: -global_mean[l]),
               counts=probe.cnt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    print(f"\n  adjacent-layer cosine, {a.n_prompts} GSM8K-{a.split} prompts, L={L}")
    print("  bucket (mask ratio high -> low):")
    for b in range(a.n_buckets):
        top = order[b][:8]
        print(f"    b{b}: most redundant {top}   cos[{top[0]}]={mat[b][top[0]]:.4f}")
    print(f"  global ranking : {res['global_order'][:12]}")
    print(f"  global cosine  : " + " ".join(f"{l}:{global_mean[l]:.3f}"
                                            for l in res['global_order'][:8]))
    print(f"  written {a.out}")


if __name__ == "__main__":
    main()
