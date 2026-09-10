"""Phase 1 A1 — the state x depth tolerance map, Family A (`results/phase1/PREREG.md`).

The generation grids answered *whether* a depth cut pays. They could not answer *where*
tolerance lives: a 300-problem accuracy has a 3-point floor on Family A, and every regime test
applied ONE skip set to two halves of the denoising (`../phase0.4/RESULTS.md` Deviation 5). This
measures per layer per state on thousands of committed tokens, teacher-forced on stored canvases,
so nothing is generated and the floor does not apply.

**What is stored, and what is not.** PREREG §1 asks for the canvas, the cache, and the full-depth
logits. The cache is not storable: LLaDA's DualCache is ~370 MB per canvas (32 layers x 2 x ~700
positions x 32 heads x 128 dims x 2 B) and the logits are 8 MB (32 positions x 126 k vocab), so
5 000 canvases would be terabytes. Only the token canvas and its block bounds are kept; the cache
and the reference logits are **rebuilt at measurement time by one full-depth forward**, which is
what `kl_greedy.py` already does and is exact rather than approximate. Cost: one extra forward per
canvas, against the 30 the per-layer sweep needs anyway.

**One pass per r-bucket per block**, so the map is balanced in the state variable rather than in
raw pass counts -- late passes outnumber early ones roughly 2:1 at full depth
(`../phase0.35/RESULTS.md` table 1) and pooling would let them dominate every average.

    python phase1_map.py --out results/phase1/map_A.json --n-problems 128
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

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
BUCKETS = [("(0.8, 1.0]", 0.8, 1.01), ("(0.6, 0.8]", 0.6, 0.8), ("(0.4, 0.6]", 0.4, 0.6),
           ("(0.2, 0.4]", 0.2, 0.4), ("[0.0, 0.2]", -0.01, 0.2)]


def bucket_of(r):
    for i, (_, lo, hi) in enumerate(BUCKETS):
        if lo < r <= hi:
            return i
    raise ValueError(r)


def token_class(tok, tid):
    """digit / operator / other, from the tokenizer (PREREG §1)."""
    s = tok.decode([int(tid)]).strip()
    if not s:
        return "other"
    if any(c.isdigit() for c in s):
        return "digit"
    if all(not c.isalnum() for c in s):
        return "operator"
    return "other"


@torch.no_grad()
def build_bank(model, tok, prompts, a):
    """Replay full depth and keep ONE refinement pass per r-bucket per block."""
    bank = []
    for pi, text in enumerate(prompts):
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        Lp = ids.shape[1]
        x = torch.full((1, Lp + a.gen_length), MASK_ID, dtype=torch.long, device=model.device)
        x[:, :Lp] = ids
        for nb in range(a.gen_length // a.block_length):
            s = Lp + nb * a.block_length
            e = s + a.block_length
            out_full = model(x, use_cache=True)
            pkv = out_full.past_key_values
            rp = torch.zeros_like(x, dtype=torch.bool); rp[:, s:e] = True
            gm = (x == MASK_ID); gm[:, e:] = False
            x0, ti = G.get_transfer_index(out_full.logits, 0.0, "low_confidence", gm, x, None,
                                          a.threshold)
            x = torch.where(ti, x0, x)
            taken = set()
            i = 1
            while (x[:, s:e] == MASK_ID).sum() > 0 and i < a.block_length:
                r = float((x[:, s:e] == MASK_ID).float().mean())
                b = bucket_of(r)
                logits = model(x[:, s:e], past_key_values=pkv, use_cache=True,
                               replace_position=rp).logits
                mb = (x[:, s:e] == MASK_ID)
                x0b, tib = G.get_transfer_index(logits, 0.0, "low_confidence", mb, x[:, s:e],
                                                None, a.threshold)
                if b not in taken and tib.any():
                    taken.add(b)
                    bank.append(dict(x=x.clone(), s=s, e=e, rp=rp.clone(), r=r, bucket=b,
                                     step=i, block=nb, problem=pi))
                blk = torch.where(tib, x0b, x[:, s:e])
                x = torch.cat([x[:, :s], blk, x[:, e:]], dim=1)
                i += 1
        if (pi + 1) % 8 == 0:
            print(f"  bank: {pi+1}/{len(prompts)} problems, {len(bank)} canvases", flush=True)
    return bank


@torch.no_grad()
def reference(model, c, thr):
    """Rebuild the cache at full depth and return (logits, commit mask, max-prob, argmax)."""
    pkv = model(c["x"], use_cache=True).past_key_values
    lg = model(c["x"][:, c["s"]:c["e"]], past_key_values=pkv, use_cache=True,
               replace_position=c["rp"]).logits
    mb = (c["x"][:, c["s"]:c["e"]] == MASK_ID)
    res = G.get_transfer_index(lg, 0.0, "low_confidence", mb, c["x"][:, c["s"]:c["e"]], None,
                               thr, return_confidence=True)
    x0, ti, conf = res
    return pkv, lg, ti, conf, lg.argmax(-1)


@torch.no_grad()
def skip_pass(model, ctrl, c, keep, pkv):
    """One refinement pass with `keep` active, on a cache written at full depth (`01` §0)."""
    ctrl.arm(tuple(keep))
    lg = model(c["x"][:, c["s"]:c["e"]], past_key_values=pkv, use_cache=True,
               replace_position=c["rp"]).logits
    ctrl.arm(None)
    return lg


def kl_on(ref_logits, lg, m):
    p = F.log_softmax(ref_logits[m].float(), dim=-1)
    q = F.log_softmax(lg[m].float(), dim=-1)
    return float((p.exp() * (p - q)).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--calib", default="/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=128)
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--keep-first", type=int, default=1)
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--n-cands", type=int, default=16,
                    help="candidate pool for the per-bucket greedy: the cosine top-N (PREREG §2)")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    cal = json.load(open(a.calib))
    L = cal["n_layers"]
    calib_ids = cal["calib_ids"][:a.n_problems]
    q, ans = P._gsm8k("train")
    shots = "".join(P.FEWSHOT_TEMPLATE.format(q=q[i], a=ans[i]) for i in range(a.n_shot))
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": shots + f"Question: {q[i]}\nAnswer:"}],
        add_generation_prompt=True, tokenize=False) for i in calib_ids]

    t0 = time.perf_counter()
    bank = build_bank(model, tok, prompts, a)
    print(f"bank: {len(bank)} canvases from {len(prompts)} problems "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    protected = set(range(a.keep_first)) | set(range(L - max(a.keep_last, 1), L))
    cands = [l for l in cal["global_order"] if l not in protected][:a.n_cands]
    ctrl = install_skipping(model)

    # split-half by problem, so every reading below carries its own replicate (PREREG §1)
    half = {c["problem"]: (c["problem"] % 2) for c in bank}

    # ---- A1a: the per-layer x per-state map ------------------------------------------------
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))   # [kl, agree, n]
    terc = {}
    for ci, c in enumerate(bank):
        pkv, ref_lg, ti, conf, ref_arg = reference(model, c, a.threshold)
        m = ti
        if not m.any():
            continue
        cvals = conf[m]
        terc[ci] = float(cvals.mean())
        for l in range(L):
            if l in protected:
                continue
            keep = [j for j in range(L) if j != l]
            lg = skip_pass(model, ctrl, c, keep, pkv)
            k = kl_on(ref_lg, lg, m)
            ag = float((lg.argmax(-1)[m] == ref_arg[m]).float().mean())
            for key in ((c["bucket"], "all"), (c["bucket"], f"half{half[c['problem']]}")):
                slot = agg[key][l]
                slot[0] += k; slot[1] += ag; slot[2] += 1
        if (ci + 1) % 50 == 0:
            print(f"  map: {ci+1}/{len(bank)} canvases", flush=True)

    res = dict(model=a.model, L=L, n_canvases=len(bank), n_problems=len(prompts),
               buckets=[b[0] for b in BUCKETS], candidates=cands, args=vars(a),
               per_layer={f"{b}|{h}": {str(l): dict(kl=v[0]/v[2], agree=v[1]/v[2], n=v[2])
                                       for l, v in d.items()}
                          for (b, h), d in agg.items()})
    uninstall_skipping(model)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
