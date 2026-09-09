"""KL-greedy skip-set selection on S — `results/phase0.25/PREREG.md` §3.

One `identity` cell per family uses a set chosen by greedy removal instead of the cosine
ranking: at each step add the admissible layer whose removal raises mean paired
KL(full || pruned) on the tokens the sampler actually commits the least, until L_eq_req layers
are chosen. It answers whether selection quality, not budget size, is what moves the pass
count — the question Phase 0 left as the binding one.

Cost. Evaluating a candidate on full generations would be ~24 problems x 90 passes per
candidate per greedy step. Instead the full-depth run on S is snapshotted into a bank of
canvases — each `(x, s, e)` with the full-depth logits over the block and the mask of
positions that pass committed — and every candidate is one forward per canvas. With 7 steps
over ~23 admissible layers that is ~4 k forwards, minutes rather than hours.

Selection uses S only (24 problems, `ids.json`), never E; that is the whole point of S.
"""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import generate as G                                                    # noqa: E402
import profile_step as P                                                # noqa: E402
from model.modeling_llada import LLaDAModelLM                           # noqa: E402
from transformers import AutoTokenizer                                  # noqa: E402
from dllm_skip.hook import install_skipping, uninstall_skipping         # noqa: E402
from dllm_skip.depth_schedule import select_skips, max_skippable        # noqa: E402

MASK_ID = 126336


@torch.no_grad()
def build_bank(model, tok, prompts, args):
    """Replay the full-depth sampler on S and snapshot canvases mid-block.

    Each entry is the state the model saw on one refinement pass, plus the reference logits
    and the positions that pass committed. Snapshots are taken every `--stride` refinement
    passes so the bank spans the whole mask-ratio range rather than clustering at block start.
    """
    bank = []
    for text in prompts:
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        Lp = ids.shape[1]
        x = torch.full((1, Lp + args.gen_length), MASK_ID, dtype=torch.long, device=model.device)
        x[:, :Lp] = ids
        for nb in range(args.gen_length // args.block_length):
            s = Lp + nb * args.block_length
            e = s + args.block_length
            out_full = model(x, use_cache=True)                       # cache-writing pass
            pkv = out_full.past_key_values
            rp = torch.zeros_like(x, dtype=torch.bool); rp[:, s:e] = True
            gm = (x == MASK_ID); gm[:, e:] = False
            x0, ti = G.get_transfer_index(out_full.logits, 0.0, "low_confidence", gm, x,
                                          None, args.threshold)
            x = torch.where(ti, x0, x)
            i = 1
            while (x[:, s:e] == MASK_ID).sum() > 0 and i < args.block_length:
                canvas = x.clone()
                logits = model(x[:, s:e], past_key_values=pkv, use_cache=True,
                               replace_position=rp).logits
                mb = (x[:, s:e] == MASK_ID)
                x0b, tib = G.get_transfer_index(logits, 0.0, "low_confidence", mb,
                                                x[:, s:e], None, args.threshold)
                if i % args.stride == 0 and tib.any():
                    bank.append(dict(x=canvas, s=s, e=e, rp=rp.clone(),
                                     ref=logits.detach().clone(), committed=tib.clone()))
                blk = torch.where(tib, x0b, x[:, s:e])
                x = torch.cat([x[:, :s], blk, x[:, e:]], dim=1)
                i += 1
            if len(bank) >= args.max_canvases:
                return bank
    return bank


@torch.no_grad()
def mean_kl(model, ctrl, bank, keep):
    """Mean KL(full || pruned) over each canvas's committed positions."""
    tot, n = 0.0, 0
    for b in bank:
        # the cache must come from a full-depth pass: the rule is cache writes stay full depth
        ctrl.arm(None)
        pkv = model(b["x"], use_cache=True).past_key_values
        ctrl.arm(keep)
        lg = model(b["x"][:, b["s"]:b["e"]], past_key_values=pkv, use_cache=True,
                   replace_position=b["rp"]).logits
        m = b["committed"]
        if not m.any():
            continue
        p = F.log_softmax(b["ref"][m].float(), dim=-1)
        q = F.log_softmax(lg[m].float(), dim=-1)
        tot += float((p.exp() * (p - q)).sum(-1).mean())
        n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--ids", default="/home1/hyunhoko/DLLM/results/phase0.25/ids.json")
    ap.add_argument("--k", type=int, required=True, help="layers to select (L_eq_req)")
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--keep-first", type=int, default=1)
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-canvases", type=int, default=96)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    S = json.load(open(a.ids))["S"]
    q, ans = P._gsm8k("train")
    shots = "".join(P.FEWSHOT_TEMPLATE.format(q=q[i], a=ans[i]) for i in range(a.n_shot))
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": shots + f"Question: {q[i]}\nAnswer:"}],
        add_generation_prompt=True, tokenize=False) for i in S]

    t0 = time.perf_counter()
    bank = build_bank(model, tok, prompts, a)
    print(f"bank: {len(bank)} canvases from {len(S)} S problems  "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    L = len(P.find_blocks(model)[1])
    ctrl = install_skipping(model)
    ceiling = max_skippable(L, a.keep_first, a.keep_last, True)
    assert a.k <= ceiling, f"k={a.k} exceeds the structural ceiling {ceiling}"

    chosen, trace = [], []
    protected = set(range(a.keep_first)) | set(range(L - max(a.keep_last, 1), L))
    for step in range(a.k):
        cands = [l for l in range(L) if l not in protected and l not in chosen
                 and not any(abs(l - c) == 1 for c in chosen)]
        scores = {}
        for l in cands:
            keep = [j for j in range(L) if j not in chosen + [l]]
            scores[l] = mean_kl(model, ctrl, bank, keep)
        best = min(scores, key=scores.get)
        chosen.append(best)
        trace.append(dict(step=step + 1, chosen=best, kl=scores[best],
                          all_scores={str(k): v for k, v in sorted(scores.items())}))
        print(f"  step {step+1}: +layer {best:2d}  KL={scores[best]:.5f}  "
              f"set={sorted(chosen)}", flush=True)
    uninstall_skipping(model)

    cal = json.load(open("/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json"))
    cosine_set = list(select_skips(cal["global_order"], a.k, L, a.keep_first, a.keep_last, True))
    res = dict(model=a.model, L=L, k=a.k, selection_set="S", n_S=len(S),
               n_canvases=len(bank), args=vars(a),
               kl_greedy_set=sorted(chosen), cosine_set=cosine_set,
               overlap=sorted(set(chosen) & set(cosine_set)), trace=trace)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  KL-greedy set : {sorted(chosen)}")
    print(f"  cosine set    : {cosine_set}")
    print(f"  overlap       : {res['overlap']}  ({len(res['overlap'])}/{a.k})")
    print(f"  written {a.out}")


if __name__ == "__main__":
    main()
