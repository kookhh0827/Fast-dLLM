"""Phase 2: the teacher-forced commit test (TF test), `results/phase2/PREREG.md` section 1.1.

For each canvas of the Family A Phase 1 bank, the script runs one refinement pass with the bf16 model and one
with the compressed model. Both passes use the same cache: the bf16 full-depth forward over the canvas, as in
Phase 1. For every masked position, the script stores the confidence (float64, the sampler's own softmax) and the
argmax of both models in `<out>/positions.npz`. The reader script computes J, A and tau'.

Compression kinds:
  depth   identity on a skip set (the control check: the KL-greedy set of 7 layers), one model, skip hooks
  gptq    a GPTQ checkpoint as expanded weights (gptq_dequant.load_into), a second model copy
  rtn     round-to-nearest at load time (gptq_dequant.rtn_quantize), a second model copy

    python phase2_tftest.py --comp depth --skip 1,3,5,9,11,14,16 --out <dir>
    python phase2_tftest.py --comp gptq --gptq <snapshot> --out <dir>
    python phase2_tftest.py --comp rtn --bits 4 --out <dir>
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import phase1_map as M                                                  # noqa: E402
import gptq_dequant as Q                                                # noqa: E402
from model.modeling_llada import LLaDAModelLM                           # noqa: E402
from dllm_skip.hook import install_skipping                             # noqa: E402


def load(model_id, dev):
    return LLaDAModelLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(dev).eval()


def weight_errors(ref, cmp):
    """Relative weight error ||W_q - W|| / ||W|| of each block linear."""
    out = {}
    with torch.no_grad():
        for l, (b0, b1) in enumerate(zip(ref.model.transformer.blocks, cmp.model.transformer.blocks)):
            for name in Q.LINEARS:
                W = getattr(b0, name).weight.float()
                Wq = getattr(b1, name).weight.float()
                out[f"{l}.{name}"] = float((Wq - W).norm() / W.norm())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--bank-state", default="/scratch2/hyunhoko/tmp/phase1/map_A.state.pt")
    ap.add_argument("--comp", required=True, choices=["depth", "gptq", "rtn"])
    ap.add_argument("--skip", default="1,3,5,9,11,14,16", help="depth: the skip set")
    ap.add_argument("--gptq", default=None, help="gptq: the snapshot directory")
    ap.add_argument("--offset", type=int, default=1, help="gptq: zero-point offset (D_gptq_check.txt)")
    ap.add_argument("--bits", type=int, default=None, help="rtn: bit width")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    ref = load(a.model, dev)
    L = len(ref.model.transformer.blocks)
    meta = dict(comp=a.comp, model=a.model, threshold=a.threshold, gpu=torch.cuda.get_device_name(0),
                fork=subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(), args=vars(a))
    if a.comp == "depth":
        skip = sorted(int(x) for x in a.skip.split(","))
        keep = [l for l in range(L) if l not in skip]
        ctrl = install_skipping(ref)
        meta["skip"] = skip
    else:
        cmp = load(a.model, dev)
        if a.comp == "gptq":
            Q.load_into(cmp, a.gptq, a.offset)
            meta["gptq"] = a.gptq
        else:
            meta["rtn_errors_at_load"] = Q.rtn_quantize(cmp, a.bits)
            meta["bits"] = a.bits
        meta["weight_errors"] = weight_errors(ref, cmp)
        e = list(meta["weight_errors"].values())
        meta["mean_rel_weight_error"] = float(np.mean(e))
        print(f"mean relative weight error over {len(e)} block linears: {meta['mean_rel_weight_error']:.5f}", flush=True)

    bank = torch.load(a.bank_state, weights_only=False)["bank"]
    if a.limit:
        bank = bank[:a.limit]
    rows = {k: [] for k in ("canvas", "pos", "problem", "half", "bucket", "conf_ref", "arg_ref", "conf_q", "arg_q")}
    t0 = time.perf_counter()
    with torch.no_grad():
        for ci, c in enumerate(bank):
            if a.comp == "depth":
                ctrl.arm(None)
            pkv, lg, _ti, conf, arg = M.reference(ref, c, a.threshold)
            s, e = c["s"], c["e"]
            mask = (c["x"][0, s:e] == M.MASK_ID)
            pos = mask.nonzero().squeeze(-1)
            if a.comp == "depth":
                lq = M.skip_pass(ref, ctrl, c, keep, pkv)[0]
            else:
                lq = cmp(c["x"][:, s:e], past_key_values=pkv, use_cache=True, replace_position=c["rp"]).logits[0]
            pq = torch.softmax(lq.double(), -1)
            cq, aq = pq.max(-1)
            n = len(pos)
            rows["canvas"].append(np.full(n, ci, np.int32))
            rows["pos"].append(pos.cpu().numpy().astype(np.int16))
            rows["problem"].append(np.full(n, c["problem"], np.int32))
            rows["half"].append(np.full(n, c["problem"] % 2, np.int8))
            rows["bucket"].append(np.full(n, c["bucket"], np.int8))
            rows["conf_ref"].append(conf[0][pos].double().cpu().numpy())
            rows["arg_ref"].append(arg[0][pos].cpu().numpy().astype(np.int32))
            rows["conf_q"].append(cq[pos].cpu().numpy())
            rows["arg_q"].append(aq[pos].cpu().numpy().astype(np.int32))
            if (ci + 1) % 500 == 0:
                print(f"  {ci + 1}/{len(bank)} canvases ({time.perf_counter() - t0:.0f} s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "positions.npz"), **{k: np.concatenate(v) for k, v in rows.items()})
    meta["n_canvases"] = len(bank)
    meta["n_positions"] = int(sum(len(x) for x in rows["pos"]))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"written {a.out}/positions.npz ({meta['n_positions']} masked positions, {len(bank)} canvases)")


if __name__ == "__main__":
    main()
