"""Phase 1b A2 / A3 / A6 on Family A's Phase 1 bank (`results/phase1b/PREREG.md`). The bank is read from
the Phase 1 state file, never rebuilt; the reference and the single-layer skip passes are the Phase 1
functions, so every number is on the same canvases as `results/phase1/tok_A`.

  A2  per canvas, per single-skipped layer l in 1..30: (i) the mean shift of max-probability on the full
      model's committed positions, (ii) the Spearman correlation between the full and skip-l max-probability
      over ALL masked positions of the block (the confidence ranking a threshold sampler uses);
  A3  every argmax flip on a committed position: whether the new token is the full model's second choice,
      the full model's margin p1 - p2, the position in the block, the token class;
  A6  the 128 calibration problems generated at full depth (tau 0.9, DualCache, gen 256, block 32, 5-shot,
      the Phase 1 settings) and scored, so crystallisation can be read against final-answer correctness.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import phase1_map as M                                                  # noqa: E402
import profile_step as P                                                # noqa: E402
import generate as G                                                    # noqa: E402
from model.modeling_llada import LLaDAModelLM                           # noqa: E402
from transformers import AutoTokenizer                                  # noqa: E402
from dllm_skip.hook import install_skipping, uninstall_skipping         # noqa: E402

CLASSES = ["digit", "operator", "other"]


def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra -= ra.mean(); rb -= rb.mean()
    d = (ra.norm() * rb.norm())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--calib", default="/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json")
    ap.add_argument("--bank-state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-a6", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    cal = json.load(open(a.calib))
    L = cal["n_layers"]
    bank = torch.load(a.bank_state, weights_only=False)["bank"]
    if a.limit:
        bank = bank[:a.limit]
    print(f"bank: {len(bank)} canvases (not rebuilt)", flush=True)
    layers = [l for l in range(1, L - 1)]
    ctrl = install_skipping(model)

    spear = np.full((len(bank), len(layers)), np.nan, np.float32)
    dshift = np.full((len(bank), len(layers)), np.nan, np.float32)
    flips = []                     # (canvas, layer, is_second, margin, pos, cls, full_p1)
    cls_cache = {}
    t0 = time.perf_counter()
    for ci, c in enumerate(bank):
        pkv, ref_lg, ti, conf, ref_arg = M.reference(model, c, a.threshold)
        mask = (c["x"][0, c["s"]:c["e"]] == M.MASK_ID)
        com = ti[0]
        if mask.sum() < 1 or com.sum() < 1:
            continue
        pf = torch.softmax(ref_lg[0].float(), -1)
        top2 = pf.topk(2, dim=-1)
        mp_full = top2.values[:, 0]
        pos = com.nonzero().squeeze(-1)
        for t in ref_arg[0][pos].tolist():
            if t not in cls_cache:
                cls_cache[t] = CLASSES.index(M.token_class(tok, t))
        for j, l in enumerate(layers):
            lg = M.skip_pass(model, ctrl, c, [x for x in range(L) if x != l], pkv)[0].float()
            ps = torch.softmax(lg, -1)
            mp_skip = ps.max(-1).values
            if mask.sum() >= 3:
                spear[ci, j] = spearman(mp_full[mask], mp_skip[mask])
            dshift[ci, j] = float((mp_skip[pos] - mp_full[pos]).mean())
            sarg = lg.argmax(-1)
            for p in pos.tolist():
                if int(sarg[p]) != int(ref_arg[0][p]):
                    flips.append((ci, l, int(sarg[p]) == int(top2.indices[p, 1]),
                                  float(top2.values[p, 0] - top2.values[p, 1]), p,
                                  cls_cache[int(ref_arg[0][p])], float(top2.values[p, 0])))
        if (ci + 1) % 200 == 0:
            print(f"  A2/A3: {ci+1}/{len(bank)} ({(time.perf_counter()-t0)/(ci+1):.2f}s/canvas)", flush=True)
    uninstall_skipping(model)
    fl = np.array(flips, dtype=np.float64) if flips else np.zeros((0, 7))
    res = dict(layers=np.array(layers), spearman=spear, dshift=dshift, flips=fl,
               c_problem=np.array([c["problem"] for c in bank]), c_bucket=np.array([c["bucket"] for c in bank]),
               c_block=np.array([c["block"] for c in bank]))

    if not a.skip_a6:
        q, ans = P._gsm8k("train")
        shots = "".join(P.FEWSHOT_TEMPLATE.format(q=q[i], a=ans[i]) for i in range(5))
        ids = cal["calib_ids"][:max(c["problem"] for c in bank) + 1]
        correct = []
        for n, i in enumerate(ids):
            text = tok.apply_chat_template([{"role": "user", "content": shots + f"Question: {q[i]}\nAnswer:"}],
                                           add_generation_prompt=True, tokenize=False)
            inp = tok(text, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                out, _st = G.generate_with_dual_cache(model, inp, steps=256, gen_length=256, block_length=32,
                                                      temperature=0.0, remasking="low_confidence",
                                                      threshold=a.threshold)
            gen = tok.decode(out[0, inp.shape[1]:], skip_special_tokens=True)
            import stage2_runner as SR
            pred = SR.flexible_extract(gen)
            gold = ans[i].split("####")[-1].strip().replace(",", "")
            try:
                ok = pred is not None and abs(float(pred) - float(gold)) < 1e-6
            except ValueError:
                ok = False
            correct.append(int(ok))
            if (n + 1) % 32 == 0:
                print(f"  A6: {n+1}/{len(ids)} generated, accuracy so far {sum(correct)/len(correct):.3f}", flush=True)
        res["calib_correct"] = np.array(correct)
    np.savez_compressed(a.out + ".npz", **res)
    print(f"written {a.out}.npz  flips {len(flips)}")


if __name__ == "__main__":
    main()
