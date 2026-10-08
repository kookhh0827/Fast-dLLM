"""Phase 2: the teacher-forced commit test (TF test), `results/phase2/PREREG.md` section 1.1, and its variants
TF-F and TF-S (section 10, amendment A2).

Modes (`--mode`, a comma-separated list; each mode writes `positions.npz` and `meta.json`):
  std  the registered test. For each canvas of the Family A Phase 1 bank, one refinement pass with the bf16 model
       and one with the compressed model. Both passes use the same cache: the bf16 full-depth forward over the
       canvas, as in Phase 1. Written to `<out>/`.
  F    TF-F (own context). Each model runs its own full forward over the canvas (`phase1_map.reference`), so each
       model builds its own cache, then its own refinement pass. Written to `<out>/F/`.
  S    TF-S (block start). For each (problem, block) of the bank, one block-start state: the canvas with the
       current block masked again. Each model runs the cache-writing pass on it (a full forward). The positions
       are the 32 positions of the block. Written to `<out>/S/`. Before the run, the script checks 3 states
       (earlier blocks committed, current and later blocks masked) and then asserts the same for every state.
For every masked position, the script stores the confidence (float64, the sampler's own softmax) and the argmax of
both models. The reader (`scripts/analysis/phase2_tf.py`) computes J, A, the commit-count ratio and the count-matched
threshold (tau' for std, tau'_F for F, tau_w' for S).

Compression kinds:
  depth   identity on a skip set (the control check: the KL-greedy set of 7 layers), one model, skip hooks; std only
  gptq    a GPTQ checkpoint as expanded weights (gptq_dequant.load_into), a second model copy
  rtn     round-to-nearest at load time (gptq_dequant.rtn_quantize), a second model copy
  soloq   a second model copy patched by SoloQ (`soloq.bridge.patch_from_env`, from SOLOQ_PATH, with SOLOQ_WA set).
          SoloQ is private: this file holds only the call. The script checks that the bf16 model is not changed.
  none    no compressed model: the compressed columns copy the bf16 columns. This run is the partner of the A/A
          test (the bf16 reference of this run against the bf16 reference of another run).

    python phase2_tftest.py --comp depth --skip 1,3,5,9,11,14,16 --out <dir>
    python phase2_tftest.py --comp gptq --gptq <snapshot> --mode std,F,S --out <dir>
    SOLOQ_PATH=<dir> SOLOQ_WA=w4a4 python phase2_tftest.py --comp soloq --mode std,F,S --out <dir>
    python phase2_tftest.py --comp none --mode F,S --out <dir>
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
import generate as G                                                    # noqa: E402
from model.modeling_llada import LLaDAModelLM                           # noqa: E402
from dllm_skip.hook import install_skipping                             # noqa: E402
from phase2_noop_check import snapshot as structure                     # noqa: E402

KEYS = ("canvas", "pos", "problem", "half", "bucket", "conf_ref", "arg_ref", "conf_q", "arg_q")


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


class Rows:
    def __init__(self):
        self.r = {k: [] for k in KEYS}
        self.extra = {}

    def add(self, ci, pos, problem, bucket, conf_ref, arg_ref, conf_q, arg_q, **extra):
        n = len(pos)
        self.r["canvas"].append(np.full(n, ci, np.int32))
        self.r["pos"].append(pos.cpu().numpy().astype(np.int16))
        self.r["problem"].append(np.full(n, problem, np.int32))
        self.r["half"].append(np.full(n, problem % 2, np.int8))
        self.r["bucket"].append(np.full(n, bucket, np.int8))
        self.r["conf_ref"].append(conf_ref.double().cpu().numpy())
        self.r["arg_ref"].append(arg_ref.cpu().numpy().astype(np.int32))
        self.r["conf_q"].append(conf_q.double().cpu().numpy())
        self.r["arg_q"].append(arg_q.cpu().numpy().astype(np.int32))
        for k, v in extra.items():
            self.extra.setdefault(k, []).append(np.full(n, v, np.int32))

    def save(self, out, meta):
        os.makedirs(out, exist_ok=True)
        arrs = {k: np.concatenate(v) for k, v in {**self.r, **self.extra}.items()}
        np.savez_compressed(os.path.join(out, "positions.npz"), **arrs)
        meta["n_positions"] = int(len(arrs["canvas"]))
        meta["n_canvases"] = int(len(np.unique(arrs["canvas"])))
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        print(f"written {out}/positions.npz ({meta['n_positions']} positions, {meta['n_canvases']} canvases or states)",
              flush=True)


@torch.no_grad()
def block_start(model, x, s, e, thr):
    """The cache-writing pass on a block-start state: a full forward, the sampler's confidence on the block."""
    lg = model(x, use_cache=True).logits
    gm = (x == M.MASK_ID)
    gm[:, e:] = False
    _x0, _ti, conf = G.get_transfer_index(lg, 0.0, "low_confidence", gm, x, None, thr, return_confidence=True)
    return conf[0, s:e], lg[0, s:e].argmax(-1)


def block_start_states(bank):
    """One state for each (problem, block): the first bank canvas of that block, with the block masked again."""
    seen, states = set(), []
    for c in bank:
        key = (c["problem"], c["block"])
        if key in seen:
            continue
        seen.add(key)
        x = c["x"].clone()
        x[:, c["s"]:] = M.MASK_ID
        states.append(dict(x=x, s=c["s"], e=c["e"], problem=c["problem"], block=c["block"], src=c))
    return states


def check_state(st):
    """Earlier blocks committed, current and later blocks masked (the source canvas: later blocks masked too)."""
    x, s, e = st["x"], st["s"], st["e"]
    return dict(problem=int(st["problem"]), block=int(st["block"]), s=int(s), length=int(x.shape[1]),
                masked_before_s=int((x[:, :s] == M.MASK_ID).sum()),
                unmasked_from_s=int((x[:, s:] != M.MASK_ID).sum()),
                source_unmasked_after_e=int((st["src"]["x"][:, e:] != M.MASK_ID).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--bank-state", default="/scratch2/hyunhoko/tmp/phase1/map_A.state.pt")
    ap.add_argument("--comp", required=True, choices=["depth", "gptq", "rtn", "soloq", "none"])
    ap.add_argument("--mode", default="std", help="comma-separated: std, F, S")
    ap.add_argument("--skip", default="1,3,5,9,11,14,16", help="depth: the skip set")
    ap.add_argument("--gptq", default=None, help="gptq: the snapshot directory")
    ap.add_argument("--offset", type=int, default=1, help="gptq: zero-point offset (D_gptq_check.txt)")
    ap.add_argument("--bits", type=int, default=None, help="rtn: bit width")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="std and F: the first canvases; S: the first states")
    a = ap.parse_args()
    modes = [m.strip() for m in a.mode.split(",") if m.strip()]
    assert modes and all(m in ("std", "F", "S") for m in modes), modes
    if a.comp == "depth":
        assert modes == ["std"], "the depth cut has only the registered mode"

    dev = torch.device("cuda")
    torch.manual_seed(0)
    ref = load(a.model, dev)
    L = len(ref.model.transformer.blocks)
    meta = dict(comp=a.comp, model=a.model, threshold=a.threshold, gpu=torch.cuda.get_device_name(0),
                fork=subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(), args=vars(a))
    cmp = None
    if a.comp == "depth":
        skip = sorted(int(x) for x in a.skip.split(","))
        keep = [l for l in range(L) if l not in skip]
        ctrl = install_skipping(ref)
        meta["skip"] = skip
    elif a.comp in ("gptq", "rtn"):
        cmp = load(a.model, dev)
        if a.comp == "gptq":
            Q.load_into(cmp, a.gptq, a.offset)
            meta["gptq"] = a.gptq
            meta["bits"] = Q.checkpoint_bits(a.gptq)
        else:
            meta["rtn_errors_at_load"] = Q.rtn_quantize(cmp, a.bits)
            meta["bits"] = a.bits
        meta["weight_errors"] = weight_errors(ref, cmp)
        e = list(meta["weight_errors"].values())
        meta["mean_rel_weight_error"] = float(np.mean(e))
        print(f"mean relative weight error over {len(e)} block linears: {meta['mean_rel_weight_error']:.5f}", flush=True)
    elif a.comp == "soloq":
        env = {k: v for k, v in os.environ.items() if k.startswith("SOLOQ_")}
        if "SOLOQ_PATH" not in env or env.get("SOLOQ_WA") != "w4a4":
            raise SystemExit(f"--comp soloq needs SOLOQ_PATH and SOLOQ_WA=w4a4 (protocol section 5): {env}")
        cmp = load(a.model, dev)
        before = structure(ref, tensors=False)
        sys.path.insert(0, env["SOLOQ_PATH"])
        import soloq.bridge
        soloq.bridge.patch_from_env(cmp)
        after = structure(ref, tensors=False)
        same = all(before[k] == after[k] for k in ("classes", "hooks", "global_hooks", "cls_forward", "funcs"))
        if not same:
            raise SystemExit("the SoloQ call changed the bf16 model (classes, hooks or module functions)")
        meta["soloq_env"] = {k: v for k, v in env.items() if k != "SOLOQ_PATH"}
        meta["soloq_commit"] = subprocess.run(["git", "-C", env["SOLOQ_PATH"], "rev-parse", "HEAD"],
                                              capture_output=True, text=True).stdout.strip()
        print(f"[soloq] second model copy patched {meta['soloq_env']} at {meta['soloq_commit'][:7]}; "
              f"bf16 model unchanged (classes, hooks, module functions)", flush=True)

    bank = torch.load(a.bank_state, weights_only=False)["bank"]
    meta["bank_canvases"] = len(bank)

    for mode in modes:
        rows, t0 = Rows(), time.perf_counter()
        mm = dict(meta, mode=mode)
        out = a.out if mode == "std" else os.path.join(a.out, mode)
        with torch.no_grad():
            if mode in ("std", "F"):
                canv = bank[:a.limit] if a.limit else bank
                for ci, c in enumerate(canv):
                    if a.comp == "depth":
                        ctrl.arm(None)
                    pkv, _lg, _ti, conf, arg = M.reference(ref, c, a.threshold)
                    s, e = c["s"], c["e"]
                    pos = (c["x"][0, s:e] == M.MASK_ID).nonzero().squeeze(-1)
                    if a.comp == "none":
                        cq, aq = conf[0], arg[0]
                    elif mode == "F":
                        _p, _l, _t, cq, aq = M.reference(cmp, c, a.threshold)        # its own cache
                        cq, aq = cq[0], aq[0]
                    else:
                        if a.comp == "depth":
                            lq = M.skip_pass(ref, ctrl, c, keep, pkv)[0]
                        else:
                            lq = cmp(c["x"][:, s:e], past_key_values=pkv, use_cache=True,
                                     replace_position=c["rp"]).logits[0]
                        cq, aq = torch.softmax(lq.double(), -1).max(-1)
                    rows.add(ci, pos, c["problem"], c["bucket"], conf[0][pos], arg[0][pos], cq[pos], aq[pos])
                    if (ci + 1) % 500 == 0:
                        print(f"  {mode}: {ci + 1}/{len(canv)} canvases ({time.perf_counter() - t0:.0f} s)", flush=True)
            else:
                states = block_start_states(bank)
                checks = [check_state(st) for st in states]
                print("  TF-S check, first 3 states:", flush=True)
                for ch in checks[:3]:
                    print(f"    {ch}", flush=True)
                bad = [ch for ch in checks if ch["masked_before_s"] or ch["unmasked_from_s"] or ch["source_unmasked_after_e"]]
                if bad:
                    raise SystemExit(f"TF-S check failed on {len(bad)} of {len(checks)} states, first: {bad[0]}")
                print(f"  TF-S check: all {len(checks)} states have earlier blocks committed and the current and "
                      f"later blocks masked", flush=True)
                mm["s_check_first3"] = checks[:3]
                mm["n_states"] = len(states)
                if a.limit:
                    states = states[:a.limit]
                pos = torch.arange(states[0]["e"] - states[0]["s"], device=dev)
                for si, st in enumerate(states):
                    cr, ar = block_start(ref, st["x"], st["s"], st["e"], a.threshold)
                    cq, aq = (cr, ar) if a.comp == "none" else block_start(cmp, st["x"], st["s"], st["e"], a.threshold)
                    rows.add(si, pos, st["problem"], M.bucket_of(1.0), cr, ar, cq, aq, block=st["block"])
                    if (si + 1) % 200 == 0:
                        print(f"  S: {si + 1}/{len(states)} states ({time.perf_counter() - t0:.0f} s)", flush=True)
        mm["seconds"] = time.perf_counter() - t0
        rows.save(out, mm)


if __name__ == "__main__":
    main()
