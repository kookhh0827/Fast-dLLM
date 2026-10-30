"""Phase 3 gate C1 of GPTQ 4-bit with the Marlin kernel (results/phase3/PREREG.md, amendments A13, change 13; A14,
change 8; A15, change 4). LLaDA-8B-Instruct, the published GPTQ checkpoint, the Marlin linears of Block 1.

1. Random inputs. For each of the 7 types of linear in layers 0, 15 and 31: 100 inputs (standard normal, bf16) in
   calls of 32 rows (the refinement passes of Block 1, --marlin-row-max 32), and 256 inputs in one call of 256 rows.
   The last call of the 100 inputs is padded with zero rows to 32 rows, and the outputs of these rows are discarded:
   this Marlin build stops with "illegal instruction" on a call of 4 rows (job 6665517; Phase 1b found the same for
   1 row), and the cells call it only with 32 rows. The kernel is called directly. The reference is the input in float32 times the dequantized weights in float32.
   C1 passes if the relative error of each input is at most 0.01.
2. Real activations. The Marlin model decodes the first GSM8K test problems (train shots, generation 256, block 32,
   threshold rule tau 0.9, DualCache). A hook reads each Marlin call: the largest absolute input (against 65,504, the
   largest fp16 value: the kernel casts its input to fp16), the largest absolute output, and that each output is
   finite. On layers 0, 15 and 31 it also gives the relative error of each call against the float32 reference.
3. The Marlin build: the path and the sha256 of the extension, and the source commit (A15, change 4).

    python phase3_gptq_c1.py --gptq <snapshot> --out <json>
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gptq_dequant as GQ                                              # noqa: E402

FP16_MAX = 65504.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def marlin_build():
    if GQ.MARLIN_DIR not in sys.path:
        sys.path.insert(0, GQ.MARLIN_DIR)
    import marlin
    import marlin_cuda
    git = lambda *c: subprocess.run(["git", "-C", GQ.MARLIN_DIR, *c], capture_output=True, text=True).stdout.strip()
    return dict(package=marlin.__file__, extension=marlin_cuda.__file__, extension_sha256=sha256(marlin_cuda.__file__),
                source_commit=git("rev-parse", "HEAD"), source_status=git("status", "--porcelain"),
                build_log=os.path.join(GQ.MARLIN_DIR, "build.log"))


def calls_of(fn, x, rows):
    """fn on x in calls of exactly `rows` rows (the last call padded with zero rows, whose outputs are dropped)."""
    out = []
    for i in range(0, x.shape[0], rows):
        c = x[i:i + rows]
        n = c.shape[0]
        if n < rows:
            c = torch.cat([c, c.new_zeros(rows - n, c.shape[1])])
        out.append(fn(c)[:n])
    return torch.cat(out)


def rel_rows(y, ref):
    return ((y - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-12))


@torch.no_grad()
def random_check(f, layers, offset, n, dev, row_max, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    rows, ref_w = [], {}
    for l in layers:
        for name in GQ.LINEARS:
            key = f"model.transformer.blocks.{l}.{name}"
            w32 = GQ.dequant(f, key, offset).to(dev)                      # float32 (out, in)
            z = GQ.unpack_cols(f.get_tensor(key + ".qzeros")).to(dev) + offset
            m = GQ.MarlinLinear(w32.to(torch.bfloat16), None, f.get_tensor(key + ".scales").to(dev), z, row_max)
            ref_w[(l, name)] = w32
            for calls, total in ((row_max, n), (256, 256)):
                x = torch.randn(total, w32.shape[1], generator=g, device=dev, dtype=torch.bfloat16)
                y = calls_of(lambda c: m.m(c.half()).float(), x, calls)
                e = rel_rows(y, x.float() @ w32.t())
                rows.append(dict(layer=l, linear=name, rows_per_call=calls, n_inputs=total,
                                 max_rel_err=float(e.max()), mean_rel_err=float(e.mean()),
                                 finite=bool(torch.isfinite(y).all())))
            del m
    return rows, ref_w


def real_check(args, ref_w, dev):
    from transformers import AutoTokenizer
    from model.modeling_llada import LLaDAModelLM
    import generate as G
    import stage2_runner as SR
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(args.model, revision=args.revision, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    GQ.load_marlin(model, args.gptq, args.offset, row_max=args.row_max)
    stats = {}

    def hook(name_l):
        l, name = name_l

        def f(mod, inp, out):
            x = inp[0]
            if x.numel() // mod.in_features > mod.row_max:                 # the bf16 path, not Marlin
                return
            s = stats.setdefault(f"{l}.{name}", dict(calls=0, max_abs_in=0.0, max_abs_out=0.0, finite=True,
                                                    max_rel_err=None))
            s["calls"] += 1
            s["max_abs_in"] = max(s["max_abs_in"], float(x.abs().max()))
            s["max_abs_out"] = max(s["max_abs_out"], float(out.abs().max()))
            s["finite"] &= bool(torch.isfinite(out).all())
            if (l, name) in ref_w:
                xs = x.reshape(-1, mod.in_features).float()
                e = float(rel_rows(out.reshape(xs.shape[0], -1).float(), xs @ ref_w[(l, name)].t()).max())
                s["max_rel_err"] = e if s["max_rel_err"] is None else max(s["max_rel_err"], e)
        return f

    for l, blk in enumerate(model.model.transformer.blocks):
        for name in GQ.LINEARS:
            getattr(blk, name).register_forward_hook(hook((l, name)))
    ids = json.load(open(args.ids))["E"][:args.problems]
    prompts, _ = SR.build(tok, ids, 5, "test", "train")
    for p in prompts:
        inp = torch.tensor([tok(p).input_ids], device=dev)
        with torch.no_grad():
            G.generate_with_dual_cache(model, inp, steps=256, gen_length=256, block_length=32, temperature=0.0,
                                       remasking="low_confidence", threshold=0.9)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gptq", required=True)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--revision", default="08b83a6feb34df1a6011b80c3c00c7563e963b07")
    ap.add_argument("--offset", type=int, default=1)
    ap.add_argument("--layers", default="0,15,31")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--row-max", type=int, default=32, help="--marlin-row-max of the Block 1 cells")
    ap.add_argument("--ids", default="/home1/hyunhoko/DLLM/results/phase0.3/ids_test.json")
    ap.add_argument("--problems", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    import phase3_records as R
    build = marlin_build()
    print(f"Marlin: {build}", flush=True)
    layers = [int(x) for x in a.layers.split(",")]
    with safe_open(f"{a.gptq}/model.safetensors", "pt") as f:
        rows, ref_w = random_check(f, layers, a.offset, a.n, dev, a.row_max)
    for r in rows:
        print(f"  layer {r['layer']:2d} {r['linear']:9s} {r['rows_per_call']:3d} rows/call: max rel err "
              f"{r['max_rel_err']:.2e}", flush=True)
    real = real_check(a, ref_w, dev)
    worst_rand = max(r["max_rel_err"] for r in rows)
    errs = [s["max_rel_err"] for s in real.values() if s["max_rel_err"] is not None]
    worst_real = max(errs) if errs else None
    max_in = max(s["max_abs_in"] for s in real.values())
    rep = dict(gate="C1 of GPTQ 4-bit, Marlin kernel (A13, change 13; A14, change 8; A15, change 4)",
               run=R.run_record(), gptq=a.gptq, gptq_sha256=R.sha256(f"{a.gptq}/model.safetensors"),
               marlin=build, tol=a.tol, row_max=a.row_max, layers=layers,
               random=dict(max_rel_err=worst_rand, all_finite=all(r["finite"] for r in rows), rows=rows),
               real=dict(problems=a.problems, ids=a.ids, max_rel_err_layers=worst_real, max_abs_in=max_in,
                         fp16_max=FP16_MAX, max_abs_out=max(s["max_abs_out"] for s in real.values()),
                         all_finite=all(s["finite"] for s in real.values()),
                         calls=sum(s["calls"] for s in real.values()), linears=real))
    rep["passed"] = bool(worst_rand <= a.tol and rep["random"]["all_finite"] and rep["real"]["all_finite"]
                         and max_in < FP16_MAX and (worst_real is None or worst_real <= a.tol))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"C1 GPTQ Marlin: random max rel err {worst_rand:.2e}; real calls {rep['real']['calls']}, max |input| "
          f"{max_in:.1f} (fp16 max {FP16_MAX:.0f}), max rel err (layers {a.layers}) {worst_real}, all finite "
          f"{rep['real']['all_finite']} -> {'pass' if rep['passed'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
