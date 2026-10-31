"""Phase 3, gate C1 of AWQ 4-bit: the error of the format of the tinygemm int4 kernel alone (no reading; CPU).

`gptq_dequant.Int4PackLinear` gives the kernel the scale and the float zero point scale * (8 - z) of each group in
bf16, the format of aten._weight_int4pack_mm. The C1 reference uses the expanded weights (codes - z) * scale with the
scale and z in float32, then a bf16 copy. This script computes, from the AWQ artifact, the weights in the kernel's
format (codes - 8) * bf16(scale) + bf16(scale * (8 - z)), and the relative error ||x Wk^T - x Wref^T|| / ||x Wref^T||
of 100 random inputs (standard normal, bf16, seed 0, the inputs of C1) for each linear of layers 0, L/2 - 1 and L - 1,
with no kernel. If these values match the C1 rows of the kernel, the C1 error comes from the format, not from the
arithmetic of the kernel. It also gives |8 - z| per linear: a zero point far from 8 makes a large float zero, and its
bf16 rounding is the same for the 128 inputs of the group.

    python phase3_awq_format.py --art <awq4.safetensors> --layers 0,13,27 --out <file.json>
"""
import argparse, json, os
import torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    f = safe_open(a.art, "pt")
    meta = json.load(open(a.art.replace(".safetensors", ".json")))
    layers = [int(x) for x in a.layers.split(",")]
    rows = []
    for l in layers:
        names = sorted({k[len(f"{l}."):-len(".codes")] for k in f.keys() if k.startswith(f"{l}.") and k.endswith(".codes")})
        for name in names:
            c, s, z = (f.get_tensor(f"{l}.{name}.{t}").float() for t in ("codes", "scales", "zeros"))
            out_f, in_f = c.shape
            cg = c.reshape(out_f, in_f // 128, 128)
            wref = ((cg - z[..., None]) * s[..., None]).reshape(out_f, in_f).bfloat16().float()
            wk = ((cg - 8) * s.bfloat16().float()[..., None]
                  + (s * (8 - z)).bfloat16().float()[..., None]).reshape(out_f, in_f)
            g = torch.Generator().manual_seed(0)
            x = torch.randn(a.n, in_f, generator=g).bfloat16().float()
            r = x @ wref.t()
            e = (x @ wk.t() - r).norm(dim=-1) / r.norm(dim=-1)
            rows.append(dict(layer=l, linear=name, max_rel_err=float(e.max()), mean_rel_err=float(e.mean()),
                             mean_abs_8_minus_z=float((8 - z).abs().mean()), max_abs_8_minus_z=float((8 - z).abs().max())))
            print(f"layer {l:2d} {name:24s} format error max {e.max():.4f} mean {e.mean():.4f}; |8 - z| mean "
                  f"{(8 - z).abs().mean():.2f}", flush=True)
    rep = dict(artifact=a.art, model=meta.get("model"), model_revision=meta.get("model_revision"), n_inputs=a.n,
               inputs="standard normal, bf16, torch.Generator seed 0 (CPU)", layers=layers,
               max_rel_err=max(r["max_rel_err"] for r in rows), rows=rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as fo:
        json.dump(rep, fo, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(f"largest format error {rep['max_rel_err']:.4f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
