"""Layer-redundancy calibration, Family B — `01_experiment_plan.md` §1, `profile_redundancy`.

Family A's `v1/llada/calibrate_depth.py` for Fast-dLLM v2 7B. Same quantity: cos(h_in, h_out)
per decoder layer, on the **masked positions of the pass the sampler is about to commit
from**, bucketed by mask ratio. A layer whose output points where its input already pointed
changed little, so a high cosine means high redundancy, and the ranking is what
`depth_schedule.select_skips` then turns into a skip set under the non-adjacency and
first/last rules.

Why this file had to exist. Stage 2's Family B cells ran without it: no Family B calibration
existed, so `stage2_runner_v2.py` fell back to `order = range(1, L-1)`, which under
non-adjacency is exactly the naive {1, 3, 5, ...} set that `PREREG.md` §3 excludes by name.
Every Family B whole-layer cell of the first grid is therefore on the excluded set, and its
STOP is a STOP on those sets rather than on the family (`RESULTS.md`, Deviations 1).

**The calibration set is fixed and is not ours to choose**: the same 128 GSM8K-train ids as
Family A, read from `calib_cosine.json`'s own `calib_ids`. `PREREG.md` §1.4 draws E from a
pool that excludes exactly those ids, so reusing them keeps E disjoint from every input to
skip-set selection in both families. Drawing a fresh calibration set here would silently
overlap E and void the pairing.

Three things this deliberately does NOT do, all inherited from the Family A file:

  * It does not average over all positions -- only the masked ones of the current pass. What
    matters is what the layer does to the tokens about to be committed.
  * It does not measure cache-writing passes. Prefill and the clean-block encode run at full
    depth by rule (`01` §0), so their redundancy cannot be spent and would only dilute the
    average. They are identified by the sampler's own `cache_write` flag, not guessed.
  * It does not pick the skip set. That is `select_skips`.

Instrumentation is hooks plus the sampler's existing pass record; `generation_functions.py`
is not modified. The mask comes from a pre-hook on `embed_tokens` (the one module guaranteed
to see the pass's token ids through `__call__`; the model's own `forward` is called directly
and would not fire a hook), and the pass type comes from the `log` object the depth wrapper
already appends to before every forward.

    python calibrate_depth_v2.py --out <json>
"""
import argparse
import json
import os
import sys
import types

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import generation_functions                                             # noqa: E402,F401
import profile_step_v2 as P                                             # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM            # noqa: E402
from dllm_skip.hook import _find_layers                                 # noqa: E402
from dllm_skip.depth_schedule import DepthSchedule                      # noqa: E402

MASK_ID = 151665


class PassGate:
    """Stands in for the runner's `log` list. The depth wrapper appends one record per pass
    *before* that pass's forward, so this is where the probe learns the mask ratio and
    whether the pass writes cache -- both without a second source of truth."""

    def __init__(self, probe, n_buckets):
        self.probe, self.n_buckets, self.records = probe, n_buckets, []

    def append(self, rec):
        self.records.append(rec)
        r = float(rec.get("mask_ratio", 1.0))
        self.probe.bucket = min(int((1.0 - r) * self.n_buckets), self.n_buckets - 1)
        self.probe.enabled = not rec.get("cache_write", False)

    def __len__(self):
        return len(self.records)


class CosineProbe:
    """cos(h_in, h_out) per decoder layer, on the masked positions of the pass."""

    def __init__(self, model, n_buckets):
        self.layers, _ = _find_layers(model)
        self.n = len(self.layers)
        self.n_buckets = n_buckets
        self.sum = [[0.0] * self.n for _ in range(n_buckets)]
        self.cnt = [[0] * self.n for _ in range(n_buckets)]
        self.bucket, self.enabled, self.mask = 0, False, None
        self._in, self._h = None, []
        inner = getattr(model, "model", model)
        self.embed = inner.embed_tokens

    def _embed_pre(self, mod, args, kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        self.mask = None if ids is None else (ids == MASK_ID)
        return None

    def _pre(self, idx):
        def f(mod, args, kwargs):
            self._in = args[0] if args else kwargs.get("hidden_states")
            return None
        return f

    def _post(self, idx):
        def f(mod, args, kwargs, output):
            if not self.enabled or self.mask is None or self._in is None:
                return None
            h_out = output[0] if isinstance(output, tuple) else output
            h_in = self._in
            if h_in.shape[1] != self.mask.shape[1] or h_out.shape[1] != self.mask.shape[1]:
                return None
            sel = self.mask[0]
            a, b = h_in[0][sel].float(), h_out[0][sel].float()
            if a.numel() == 0:
                return None
            self.sum[self.bucket][idx] += F.cosine_similarity(a, b, dim=-1).mean().item()
            self.cnt[self.bucket][idx] += 1
            return None
        return f

    def __enter__(self):
        self._h.append(self.embed.register_forward_pre_hook(self._embed_pre, with_kwargs=True))
        for i, blk in enumerate(self.layers):
            self._h.append(blk.register_forward_pre_hook(self._pre(i), with_kwargs=True))
            self._h.append(blk.register_forward_hook(self._post(i), with_kwargs=True))
        return self

    def __exit__(self, *a):
        for h in self._h:
            h.remove()
        self._h.clear()

    def matrix(self):
        return [[(self.sum[b][l] / self.cnt[b][l]) if self.cnt[b][l] else float("nan")
                 for l in range(self.n)] for b in range(self.n_buckets)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    ap.add_argument("--calib-ids", default="/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json",
                    help="Family A's calibration file; its calib_ids are reused verbatim")
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--small-block-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--n-buckets", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    # the checkpoint's own class has no instrumented sampler; bind the fork's, exactly as
    # `stage2_runner_v2.py` does, so calibration and the cells run the same loop
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model)

    calib_ids = json.load(open(a.calib_ids))["calib_ids"][:a.n_prompts]
    q, _ = P._gsm8k("train")
    prompts = []
    for i in calib_ids:
        text = f"Question: {q[i]}\nAnswer:".replace("Answer:", P.GSM8K_INSTRUCTION)
        prompts.append(tok.apply_chat_template([{"role": "user", "content": text}],
                                               add_generation_prompt=True, tokenize=False))

    probe = CosineProbe(model, a.n_buckets)
    L = probe.n
    gate = PassGate(probe, a.n_buckets)
    # A k = 0 schedule, not None: the sampler then takes the identical instrumented path the
    # cells take, so `cache_write` is the sampler's own flag rather than a reconstruction.
    sched = DepthSchedule.static(L, [list(range(1, L - 1))], 0, keep_first=1, keep_last=8)

    print(f"L={L}  {len(prompts)} calibration prompts (ids from {a.calib_ids})", flush=True)
    with torch.no_grad(), probe:
        for n, text in enumerate(prompts):
            ids = tok([text], return_tensors="pt").input_ids.to(dev)
            gate.records.clear()
            model.mdm_sample(ids, tokenizer=tok, block_size=a.block_size,
                             small_block_size=a.small_block_size,
                             max_new_tokens=a.max_new_tokens, mask_id=MASK_ID,
                             min_len=ids.shape[1],
                             seq_len=torch.tensor([ids.shape[1]], device=dev),
                             use_block_cache=False, threshold=a.threshold,
                             schedule=sched, controller=None, log=gate)
            probe.enabled = False
            if (n + 1) % 8 == 0:
                print(f"  {n+1}/{len(prompts)} prompts", flush=True)

    mat = probe.matrix()
    order = [sorted(range(L), key=lambda l, b=b: -mat[b][l]) for b in range(a.n_buckets)]
    glob = [sum(mat[b][l] for b in range(a.n_buckets)) / a.n_buckets for l in range(L)]
    res = dict(model=a.model, n_layers=L, args=vars(a), gpu=torch.cuda.get_device_name(0),
               calib_ids=calib_ids, cosine=mat, skip_order=order, global_cosine=glob,
               global_order=sorted(range(L), key=lambda l: -glob[l]), counts=probe.cnt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    print(f"\n  adjacent-layer cosine, {len(prompts)} GSM8K-train prompts, L={L}")
    for b in range(a.n_buckets):
        top = order[b][:8]
        print(f"    b{b}: most redundant {top}   cos[{top[0]}]={mat[b][top[0]]:.4f}")
    print(f"  global ranking : {res['global_order'][:12]}")
    print("  global cosine  : " + " ".join(f"{l}:{glob[l]:.3f}" for l in res['global_order'][:8]))
    print(f"  written {a.out}")


if __name__ == "__main__":
    main()
