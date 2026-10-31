"""Phase 3, gate C of SoloQ W4A4: check C1-S, the application check (amendment A11, change 2), on LLaDA-8B-Instruct
and on the Qwen2-like models (Fast-dLLM v2 7B; the module names of `phase3_compress.Arch`).

The SoloQ cells call the authors' code, so gate C1 (a comparison with a reference implementation) does not apply.
C1-S checks in its place that the patch reaches each linear layer and acts:
  1. the number of linear layers that SoloQ patches, and the names of the linear layers that stay in bf16;
  2. for 100 random inputs (standard normal, bf16) to each type of patched layer (in 3 layers), the relative error
     ||y - y_bf16|| / ||y_bf16|| of the patched output against the bf16 output. It must be above 0; no upper limit;
  3. the SoloQ variant that SOLOQ_WA=w4a4 selects (SoloQ-C: the codebook; SoloQ-N: NVFP4), from the configuration
     of `soloq.bridge.config_from_env`.
SoloQ is private: the fork holds only this check. The code is imported from SOLOQ_PATH (outside the fork and outside
the project repository), the same call as `stage2_runner.py`.

    SOLOQ_PATH=/home1/hyunhoko/SoloQ SOLOQ_WA=w4a4 python phase3_soloq_c1s.py --out <file.json>
    SOLOQ_PATH=... SOLOQ_WA=w4a4 python phase3_soloq_c1s.py --model Efficient-Large-Model/Fast_dLLM_v2_7B \
        --revision 0661abf5f9f0ee338970d091052a26c8efa51974 --out <file.json>
The model is loaded as the runners load it (phase3_compress.load_model), and the patch is the call of the runners
(soloq.bridge.patch_from_env). The bf16 output includes the bias of the layer, if any.
"""
import argparse
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_compress import ROLES, Arch, load_model                             # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    assert os.environ.get("SOLOQ_PATH") and os.environ.get("SOLOQ_WA") == "w4a4", "set SOLOQ_PATH and SOLOQ_WA=w4a4"
    dev = torch.device("cuda")
    model = load_model(a.model, dev, a.revision)
    A = Arch(model)
    L = len(A.blocks)
    layers = (0, L // 2 - 1, L - 1)
    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    full = {id(m): n for n, m in linears.items()}
    lname = {(l, r): full[id(A.lin(A.blocks[l], r))] for l in layers for r in ROLES}
    ref = {}
    for l in layers:
        for r in ROLES:
            m = linears[lname[(l, r)]]
            ref[(l, r)] = (m.weight.data.clone(), None if m.bias is None else m.bias.data.clone())

    sys.path.insert(0, os.environ["SOLOQ_PATH"])
    import soloq.bridge
    cfg = soloq.bridge.config_from_env()
    state = soloq.bridge.patch_from_env(model)
    patched = sorted(state.original_weights) if hasattr(state, "original_weights") else []
    hooked = sorted(n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m._forward_pre_hooks)
    names = sorted(linears)

    def same(n, k):  # SoloQ names the layers from its unwrapped model, so one name can be a suffix of the other
        return n == k or n.endswith("." + k) or k.endswith("." + n)
    changed = [k for k in patched if any(same(n, k) for n in names)]
    stay = [n for n in names if not any(same(n, k) for k in patched)]

    g = torch.Generator(device=dev).manual_seed(0)
    rows = []
    for (l, r), (W, b) in ref.items():
        m = dict(model.named_modules())[lname[(l, r)]]
        x = torch.randn(1, a.n, m.in_features, generator=g, device=dev, dtype=torch.bfloat16)
        y = m(x).float()[0]
        y0 = F.linear(x, W, b).float()[0]
        rel = (y - y0).norm(dim=1) / y0.norm(dim=1).clamp(min=1e-12)
        rows.append(dict(layer=l, linear=A.names[r], min_rel_err=float(rel.min()), mean_rel_err=float(rel.mean()),
                         max_rel_err=float(rel.max())))
    variant = "SoloQ-N (NVFP4)" if cfg and cfg.get("nvfp4") else "SoloQ-C (codebook)"
    rep = dict(
        check="C1-S (amendment A11, change 2)", model=a.model, model_revision=a.revision, n_inputs=a.n, layers=list(layers),
        soloq_commit=subprocess.run(["git", "-C", os.environ["SOLOQ_PATH"], "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(),
        soloq_env={k: v for k, v in os.environ.items() if k.startswith("SOLOQ_") and k != "SOLOQ_PATH"},
        config=cfg, variant=variant, simulated=not os.environ.get("SOLOQ_KERNEL_ARTIFACT"),
        n_linear=len(names), n_patched=len(patched), n_with_activation_hook=len(hooked),
        n_patched_found_in_model=len(changed), stay_bf16=stay, rows=rows,
        min_rel_err=min(x["min_rel_err"] for x in rows),
        passed=len(patched) > 0 and len(changed) == len(patched) and all(x["min_rel_err"] > 0 for x in rows))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as fo:
        json.dump(rep, fo, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(f"C1-S SoloQ W4A4: {variant}; {len(patched)} of {len(names)} linear layers patched "
          f"({len(hooked)} with an activation hook); in bf16: {stay}; relative error against bf16 "
          f"{rep['min_rel_err']:.3e} (smallest) to {max(x['max_rel_err'] for x in rows):.3e} (largest) -> "
          f"{'pass' if rep['passed'] else 'FAIL'}", flush=True)
    sys.exit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
