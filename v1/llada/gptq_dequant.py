"""Load a published GPTQ checkpoint of LLaDA-8B-Instruct into the fork's bf16 `LLaDAModelLM` by dequantising its
4-bit block linears (`results/phase1b/D_checkpoint.md`, PREREG D). The accuracy, NFE and confidence of the 4-bit
weights are measured exactly this way; the matmul runs in bf16, so any speed measured on it is the bf16 kernel's and
is reported as such (the projection is labelled separately).

Packing (AutoGPTQ / GPTQModel, bits 4, group 128, desc_act false): `qweight` int32 (in/8, out), eight nibbles per
word along the input axis; `qzeros` int32 (groups, out/8) packed along the output axis; `scales` (groups, out).
W[out, in] = scales[g(in), out] * (q[out, in] - (z[g(in), out] + offset)), offset 1 for the v1 on-disk format
(zeros stored minus one), 0 for v2. The checkpoint's two config files disagree on the format, so `check` decides it
from the weights: the right offset leaves no systematic bias against the bf16 original.

    python gptq_dequant.py check --gptq <snapshot> --model GSAI-ML/LLaDA-8B-Instruct
"""
import argparse
import sys

import torch
from safetensors import safe_open

LINEARS = ["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"]


def unpack_rows(q):          # (n, cols) int32 -> (n*8, cols) nibbles, along rows
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    x = (q.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF
    return x.reshape(q.shape[0] * 8, q.shape[1])


def unpack_cols(q):          # (rows, n) int32 -> (rows, n*8), along columns
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    x = (q.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF
    return x.reshape(q.shape[0], q.shape[1] * 8)


def dequant(f, prefix, offset, group=128):
    qw = f.get_tensor(prefix + ".qweight"); qz = f.get_tensor(prefix + ".qzeros")
    sc = f.get_tensor(prefix + ".scales").float(); gi = f.get_tensor(prefix + ".g_idx").long()
    iw = unpack_rows(qw).float()                  # (in, out)
    z = unpack_cols(qz).float() + offset          # (groups, out)
    n_in, n_out = iw.shape
    assert torch.equal(gi, torch.arange(n_in) // group), "g_idx is not trivial (act-order checkpoint)"
    w = sc[gi] * (iw - z[gi])                     # (in, out)
    return w.t().contiguous()                     # (out, in)


def load_into(model, snapshot, offset):
    blocks = model.model.transformer.blocks
    with safe_open(f"{snapshot}/model.safetensors", "pt") as f, torch.no_grad():
        for l, blk in enumerate(blocks):
            for name in LINEARS:
                lin = getattr(blk, name)
                w = dequant(f, f"model.transformer.blocks.{l}.{name}", offset)
                assert w.shape == lin.weight.shape, (l, name, w.shape, lin.weight.shape)
                lin.weight.copy_(w.to(lin.weight.dtype))
    return model


class Int4PackLinear(torch.nn.Module):
    """A block linear on torch's tinygemm int4 kernel (`aten._weight_int4pack_mm`, group 128, inner_k_tiles 8):
    W = (q - 8) * scale + zero with a float zero, so a GPTQ weight scale * (q - z) maps exactly to zero = scale * (8 - z).
    The 4-bit bytes are what the matmul reads, so wall-clock on this module is a real 4-bit measurement."""

    def __init__(self, qint, scales, zeros, out_f, in_f, dtype, group=128):
        super().__init__()
        self.in_features, self.out_features, self.group = in_f, out_f, group
        u8 = ((qint[:, ::2] << 4) | qint[:, 1::2]).to(torch.uint8)             # (out, in/2)
        self.register_buffer("packed", torch.ops.aten._convert_weight_to_int4pack(u8.contiguous(), 8))
        zf = scales * (8 - zeros)                                               # float zero, (groups, out)
        sz = torch.stack([scales, zf], -1).contiguous()                        # (groups, out, 2), the kernel's layout
        self.register_buffer("scales_and_zeros", sz.to(dtype))

    def forward(self, x):
        shp = x.shape
        y = torch.ops.aten._weight_int4pack_mm(x.reshape(-1, self.in_features).contiguous(), self.packed,
                                               self.group, self.scales_and_zeros)
        return y.reshape(*shp[:-1], self.out_features)


def int4pack_linear(f, prefix, offset, dev, dtype, group=128):
    qw = f.get_tensor(prefix + ".qweight"); qz = f.get_tensor(prefix + ".qzeros")
    sc = f.get_tensor(prefix + ".scales").float(); gi = f.get_tensor(prefix + ".g_idx").long()
    iw = unpack_rows(qw).t().contiguous()                                        # (out, in) nibbles
    z = unpack_cols(qz).float() + offset                                         # (groups, out)
    out_f, in_f = iw.shape
    assert torch.equal(gi, torch.arange(in_f) // group)
    return Int4PackLinear(iw.to(dev).int(), sc.to(dev), z.to(dev), out_f, in_f, dtype, group)


def load_int4pack(model, snapshot, offset):
    """Replace every block linear by an Int4PackLinear (frees the bf16 weight)."""
    blocks = model.model.transformer.blocks
    dev = next(model.parameters()).device; dtype = next(model.parameters()).dtype
    with safe_open(f"{snapshot}/model.safetensors", "pt") as f, torch.no_grad():
        for l, blk in enumerate(blocks):
            for name in LINEARS:
                lin = getattr(blk, name)
                assert lin.bias is None
                setattr(blk, name, int4pack_linear(f, f"model.transformer.blocks.{l}.{name}", offset, dev, dtype))
    torch.cuda.empty_cache()
    return model


MARLIN_DIR = "/project2/ppanda_1750/kook/dllm/kernels/marlin"


class MarlinLinear(torch.nn.Module):
    """A block linear with two paths, dispatched by the number of rows in the call (amendment (f) item 2):
    <= row_max rows (a DualCache refinement pass: 32 positions) -> the Marlin fp16 x int4 kernel on the 4-bit bytes,
    input cast to fp16 and the output cast back; more rows (the block-start pass over the whole canvas, compute-bound)
    -> a bf16 copy of the expanded 4-bit weights, i.e. exactly the E cells' numerics."""

    def __init__(self, weight_bf16, qint, scales, zeros, row_max=64):
        super().__init__()
        if MARLIN_DIR not in sys.path:
            sys.path.insert(0, MARLIN_DIR)
        import marlin
        out_f, in_f = weight_bf16.shape
        assert torch.all(zeros == 8), "Marlin is symmetric with zero point 8; this checkpoint is not"
        self.in_features, self.out_features, self.row_max = in_f, out_f, row_max
        self.register_buffer("w_bf16", weight_bf16)
        lin = torch.nn.Linear(in_f, out_f, bias=False, device=weight_bf16.device, dtype=torch.half)
        lin.weight.data.copy_(weight_bf16.float().half())
        self.m = marlin.Layer(in_f, out_f, groupsize=128).to(weight_bf16.device)
        self.m.pack(lin, scales.t().contiguous().half())              # (out, groups) as Layer.pack expects
        del lin

    def forward(self, x):
        rows = x.numel() // self.in_features
        if rows > self.row_max:
            return torch.nn.functional.linear(x, self.w_bf16.to(x.dtype))
        return self.m(x.half()).to(x.dtype)


class Int4DispatchLinear(torch.nn.Module):
    """The torch int4 kernel with the same row dispatch as MarlinLinear (a bf16 copy of the expanded weights on calls
    with > row_max rows), so the reference kernel and Marlin see identical block-start caches (amendment (g) (b))."""

    def __init__(self, weight_bf16, kernel, row_max=64):
        super().__init__()
        self.in_features, self.row_max = weight_bf16.shape[1], row_max
        self.register_buffer("w_bf16", weight_bf16)
        self.k = kernel

    def forward(self, x):
        if x.numel() // self.in_features > self.row_max:
            return torch.nn.functional.linear(x, self.w_bf16.to(x.dtype))
        return self.k(x)


def swap_kernel(model, snapshot, offset, kind, row_max=64):
    """Replace each block linear (plain Linear with expanded weights, or a dispatch module) by `kind` in {int4, marlin},
    keeping the expanded bf16 weight as the > row_max path."""
    blocks = model.model.transformer.blocks
    dev = next(model.parameters()).device
    with safe_open(f"{snapshot}/model.safetensors", "pt") as f, torch.no_grad():
        for l, blk in enumerate(blocks):
            for name in LINEARS:
                cur = getattr(blk, name)
                w = cur.weight.data if isinstance(cur, torch.nn.Linear) else cur.w_bf16
                prefix = f"model.transformer.blocks.{l}.{name}"
                if kind == "int4":
                    new = Int4DispatchLinear(w, int4pack_linear(f, prefix, offset, dev, torch.bfloat16), row_max)
                else:
                    z = unpack_cols(f.get_tensor(prefix + ".qzeros")).to(dev) + offset
                    new = MarlinLinear(w, None, f.get_tensor(prefix + ".scales").to(dev), z, row_max)
                setattr(blk, name, new)
    torch.cuda.empty_cache()
    return model


def load_marlin(model, snapshot, offset, row_max=64):
    blocks = model.model.transformer.blocks
    dev = next(model.parameters()).device
    with safe_open(f"{snapshot}/model.safetensors", "pt") as f, torch.no_grad():
        for l, blk in enumerate(blocks):
            for name in LINEARS:
                prefix = f"model.transformer.blocks.{l}.{name}"
                w = dequant(f, prefix, offset).to(dev, torch.bfloat16)
                z = unpack_cols(f.get_tensor(prefix + ".qzeros")).to(dev) + offset
                sc = f.get_tensor(prefix + ".scales").to(dev)
                setattr(blk, name, MarlinLinear(w, None, sc, z, row_max))
    torch.cuda.empty_cache()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "kernelcheck", "marlincheck", "gatecheck", "diag"])
    ap.add_argument("--bank-state", default="/scratch2/hyunhoko/tmp/phase1/map_A.state.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gptq", required=True)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--layers", default="0,15,31")
    ap.add_argument("--offset", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "kernelcheck":
        return kernelcheck(a)
    if a.cmd == "marlincheck":
        return marlincheck(a)
    if a.cmd == "gatecheck":
        return gatecheck(a)
    if a.cmd == "diag":
        return diag(a)
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    from model.modeling_llada import LLaDAModelLM
    ref = LLaDAModelLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    sd = {k: v for k, v in ref.state_dict().items()}
    out = {}
    with safe_open(f"{a.gptq}/model.safetensors", "pt") as f:
        for l in map(int, a.layers.split(",")):
            for name in LINEARS:
                key = f"model.transformer.blocks.{l}.{name}"
                W = sd[key + ".weight"].float()
                sc = f.get_tensor(key + ".scales").float()
                for off in (0, 1):
                    w = dequant(f, key, off)
                    err = w - W
                    out[f"{l}.{name}.off{off}"] = dict(rel_err=float(err.norm() / W.norm()),
                                                       mean_err_over_scale=float(err.mean() / sc.mean()),
                                                       corr=float(torch.corrcoef(torch.stack([w.flatten()[::97], W.flatten()[::97]]))[0, 1]))
        others = {}
        for k in ("model.transformer.wte.weight", "model.transformer.ff_out.weight", "model.transformer.blocks.0.attn_norm.weight"):
            t = f.get_tensor(k).float(); r = sd[k].float()
            others[k] = float((t - r).abs().max())
    for k, v in out.items():
        print(k, {kk: round(vv, 5) for kk, vv in v.items()})
    print("non-quantised tensors, max |fp16 checkpoint - bf16 original|:", others)


def kernelcheck(a):
    """GPU: int4 kernel output vs F.linear on the dequantised weight, per linear of three layers; then one DualCache
    refinement pass timed at bf16, dequantised-bf16 and int4 (32-token block, CUDA-synchronised, median of 50)."""
    import os
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model.modeling_llada import LLaDAModelLM
    dev = torch.device("cuda")
    with safe_open(f"{a.gptq}/model.safetensors", "pt") as f:
        for l in map(int, a.layers.split(",")):
            for name in LINEARS:
                key = f"model.transformer.blocks.{l}.{name}"
                w = dequant(f, key, a.offset).to(dev, torch.bfloat16)
                m = int4pack_linear(f, key, a.offset, dev, torch.bfloat16)
                x = torch.randn(32, w.shape[1], device=dev, dtype=torch.bfloat16)
                ref = torch.nn.functional.linear(x.float(), w.float()); out = m(x).float()
                print(f"  {key}: rel err {float((out - ref).norm() / ref.norm()):.2e}", flush=True)

    def timed(model):
        torch.manual_seed(0)
        x = torch.randint(0, 1000, (1, 700), device=dev)
        rp = torch.zeros_like(x, dtype=torch.bool); rp[:, 600:632] = True
        with torch.no_grad():
            pkv = model(x, use_cache=True).past_key_values
            for _ in range(10):
                model(x[:, 600:632], past_key_values=pkv, use_cache=True, replace_position=rp)
            ts = []
            for _ in range(50):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                lg = model(x[:, 600:632], past_key_values=pkv, use_cache=True, replace_position=rp).logits
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(3):
                model(x, use_cache=True)
            torch.cuda.synchronize(); tw = (time.perf_counter() - t0) / 3
        return sorted(ts)[25], tw, lg

    model = LLaDAModelLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    t_bf, w_bf, lg_bf = timed(model)
    load_into(model, a.gptq, a.offset)
    t_dq, w_dq, lg_dq = timed(model)
    load_int4pack(model, a.gptq, a.offset)
    t_q, w_q, lg_q = timed(model)
    agree = float((lg_q.argmax(-1) == lg_dq.argmax(-1)).float().mean())
    print(f"refinement pass (32 tokens, ctx 700), median ms: bf16 {1e3*t_bf:.2f}  dequant-bf16 {1e3*t_dq:.2f}  "
          f"int4 kernel {1e3*t_q:.2f}  -> int4 / bf16 = {t_q/t_bf:.3f}")
    print(f"block-start pass (700 tokens), mean ms: bf16 {1e3*w_bf:.1f}  int4 {1e3*w_q:.1f}  -> ratio {w_q/w_bf:.3f}")
    print(f"logits int4 kernel vs dequant-bf16: argmax agreement {agree:.4f}, max |diff| {float((lg_q.float()-lg_dq.float()).abs().max()):.3f}")
    print(f"memory allocated after int4 load: {torch.cuda.memory_allocated()/1e9:.2f} GB")


def marlincheck(a):
    """Amendment (f) item 2, in order: (1) per-linear output of the Marlin path vs F.linear on the expanded weights at
    32 rows; (2) THE GATE -- top-token agreement, Marlin model vs expanded-bf16 model, on every masked position of the
    block pass of the stored Phase 1 canvases (the torch int4 kernel scored 100 % on random tokens,
    D_kernel_check.txt); above 0.5 % disagreement nothing is timed; (3) pass timings, CUDA-synchronised."""
    import os
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model.modeling_llada import LLaDAModelLM
    MASK_ID = 126336
    dev = torch.device("cuda")
    with safe_open(f"{a.gptq}/model.safetensors", "pt") as f:
        allz = True
        for l in range(32):
            for name in LINEARS:
                z = unpack_cols(f.get_tensor(f"model.transformer.blocks.{l}.{name}.qzeros")) + a.offset
                allz &= bool(torch.all(z == 8))
        print(f"zero points all 8 (Marlin-eligible): {allz}", flush=True)
        for l in map(int, a.layers.split(",")):
            for name in LINEARS:
                key = f"model.transformer.blocks.{l}.{name}"
                w = dequant(f, key, a.offset).to(dev, torch.bfloat16)
                z = unpack_cols(f.get_tensor(key + ".qzeros")).to(dev) + a.offset
                m = MarlinLinear(w, None, f.get_tensor(key + ".scales").to(dev), z)
                x = torch.randn(1, 32, w.shape[1], device=dev, dtype=torch.bfloat16)
                ref = torch.nn.functional.linear(x.float(), w.float()); out = m(x).float()
                print(f"  {key}: rel err {float((out - ref).norm() / ref.norm()):.2e}", flush=True)

    model = LLaDAModelLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    load_into(model, a.gptq, a.offset)
    bank = torch.load(a.bank_state, weights_only=False)["bank"]
    if a.limit:
        bank = bank[:a.limit]

    def block_pass(c):
        with torch.no_grad():
            pkv = model(c["x"], use_cache=True).past_key_values
            return model(c["x"][:, c["s"]:c["e"]], past_key_values=pkv, use_cache=True, replace_position=c["rp"]).logits

    ref = []
    for c in bank:
        m = (c["x"][0, c["s"]:c["e"]] == MASK_ID)
        ref.append(block_pass(c)[0][m].argmax(-1).cpu())
    load_marlin(model, a.gptq, a.offset)
    agree = tot = 0
    for c, r in zip(bank, ref):
        m = (c["x"][0, c["s"]:c["e"]] == MASK_ID)
        o = block_pass(c)[0][m].argmax(-1).cpu()
        agree += int((o == r).sum()); tot += len(r)
    dis = 1 - agree / tot
    print(f"GATE top-token agreement on {len(bank)} stored canvases ({tot} masked positions): {agree/tot:.5f} "
          f"(disagreement {100*dis:.3f} %; torch int4 kernel: 100 % on random tokens) -> "
          f"{'PASS' if dis <= 0.005 else 'STOP (> 0.5 %)'}", flush=True)
    if dis > 0.005:
        return

    def timed():
        x = torch.randint(0, 1000, (1, 700), device=dev)
        rp = torch.zeros_like(x, dtype=torch.bool); rp[:, 600:632] = True
        with torch.no_grad():
            pkv = model(x, use_cache=True).past_key_values
            for _ in range(10):
                model(x[:, 600:632], past_key_values=pkv, use_cache=True, replace_position=rp)
            ts = []
            for _ in range(50):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                model(x[:, 600:632], past_key_values=pkv, use_cache=True, replace_position=rp)
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        return sorted(ts)[25]
    t_m = timed()
    model2 = LLaDAModelLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    model, model2 = model2, model
    t_b = timed()
    print(f"refinement pass (32 tokens, ctx 700), median ms: bf16 {1e3*t_b:.2f}  marlin {1e3*t_m:.2f}  -> "
          f"marlin / bf16 = {t_m/t_b:.3f}  (torch int4 kernel 1.918, D_kernel_check.txt)")


def gatecheck(a):
    """Amendment (g): on every stored Phase 1 canvas, the block pass under (ref) expanded bf16, (a) the same again
    (rerun floor), (b) the torch int4 kernel, (c) Marlin -- (b) and (c) with the bf16 copy on the block-start pass.
    Top-token disagreement against ref on committed positions (masked, ref max-prob >= 0.9) and on the other masked
    positions. GATE = (c) committed minus (a) committed, in percentage points; <= 0.5 pp -> timing runs."""
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model.modeling_llada import LLaDAModelLM
    MASK_ID = 126336
    dev = torch.device("cuda")
    model = LLaDAModelLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    load_into(model, a.gptq, a.offset)
    bank = torch.load(a.bank_state, weights_only=False)["bank"]
    if a.limit:
        bank = bank[:a.limit]

    def run_all():
        out = []
        with torch.no_grad():
            for c in bank:
                pkv = model(c["x"], use_cache=True).past_key_values
                lg = model(c["x"][:, c["s"]:c["e"]], past_key_values=pkv, use_cache=True,
                           replace_position=c["rp"]).logits[0].float()
                m = (c["x"][0, c["s"]:c["e"]] == MASK_ID)
                p = torch.softmax(lg[m], -1)
                mp, am = p.max(-1)
                out.append((am.cpu(), mp.cpu()))
        return out

    ref = run_all()
    com = [mp >= 0.9 for _, mp in ref]
    n_com = int(sum(int(x.sum()) for x in com)); n_unc = int(sum(int((~x).sum()) for x in com))

    def dis(run):
        dc = du = 0
        for (am, _), (r, _), cm in zip(run, ref, com):
            d = am != r
            dc += int((d & cm).sum()); du += int((d & ~cm).sum())
        return 100 * dc / max(n_com, 1), 100 * du / max(n_unc, 1), 100 * (dc + du) / (n_com + n_unc)

    res = {"(a) bf16 rerun": dis(run_all())}
    swap_kernel(model, a.gptq, a.offset, "int4")
    res["(b) torch int4"] = dis(run_all())
    swap_kernel(model, a.gptq, a.offset, "marlin")
    res["(c) Marlin"] = dis(run_all())
    print(f"canvases {len(bank)}; masked positions {n_com + n_unc}: committed (ref max-prob >= 0.9) {n_com}, "
          f"uncommitted {n_unc}")
    print(f"{'run':16s} {'committed %':>12s} {'uncommitted %':>14s} {'all masked %':>13s}   (top-token disagreement vs ref)")
    for k, (c_, u_, t_) in res.items():
        print(f"{k:16s} {c_:12.3f} {u_:14.3f} {t_:13.3f}")
    g = res["(c) Marlin"][0] - res["(a) bf16 rerun"][0]
    gi = res["(b) torch int4"][0] - res["(a) bf16 rerun"][0]
    print(f"GATE (amendment (g)): Marlin committed - rerun committed = {g:+.3f} pp (torch int4 under the same definition "
          f"{gi:+.3f} pp) -> {'PASS: timing runs' if g <= 0.5 else 'FAIL: no timing, projection final'}")


def diag(a):
    """Amendment (h) (a) and (b). (a) kernel-only micro-benchmark on layer 15's seven block linears at M = 1, 32, 700:
    cuBLAS bf16 (F.linear) vs the Marlin kernel call alone (preallocated fp16 input and output), with the input cast
    (bf16 -> fp16), the output allocation, the output cast (fp16 -> bf16) and the whole MarlinLinear.forward timed
    separately; CUDA events, 50 warm-up + 300 timed iterations each, per-layer sums weighted by the layer's linears.
    (b) one DualCache refinement pass (32 tokens, ctx 700) under the torch profiler, bf16 expanded vs Marlin: GEMM
    kernel CUDA time vs everything else, kernel launch counts, and any compiled / graphed regions."""
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if MARLIN_DIR not in sys.path:
        sys.path.insert(0, MARLIN_DIR)
    import marlin
    from model.modeling_llada import LLaDAModelLM
    dev = torch.device("cuda")
    ITER, WARM = 300, 50

    def ev_time(fn):
        for _ in range(WARM):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(ITER):
            s_, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s_.record(); fn(); e_.record(); torch.cuda.synchronize()
            ts.append(s_.elapsed_time(e_))
        ts.sort()
        return ts[len(ts) // 2]                                  # median ms

    print("## (a) kernel-only micro-benchmark, layer 15's linears, median ms over 300 CUDA-event-timed iterations")
    rows = []
    with safe_open(f"{a.gptq}/model.safetensors", "pt") as f:
        mods = {}
        for name in LINEARS:
            key = f"model.transformer.blocks.15.{name}"
            w = dequant(f, key, a.offset).to(dev, torch.bfloat16)
            z = unpack_cols(f.get_tensor(key + ".qzeros")).to(dev) + a.offset
            mods[name] = MarlinLinear(w, None, f.get_tensor(key + ".scales").to(dev), z, row_max=10 ** 9)
    for M in (1, 32, 700):
        tot = dict(cublas=0.0, kernel=0.0, cast_in=0.0, alloc=0.0, cast_out=0.0, wrapper=0.0)
        for name in LINEARS:
            m = mods[name]; W = m.w_bf16; in_f, out_f = W.shape[1], W.shape[0]
            x = torch.randn(1, M, in_f, device=dev, dtype=torch.bfloat16)
            xh = x.half().view(-1, in_f); C = torch.empty(M, out_f, device=dev, dtype=torch.half)
            t = dict(
                cublas=ev_time(lambda: torch.nn.functional.linear(x, W)),
                kernel=ev_time(lambda: marlin.mul(xh, m.m.B, C, m.m.s, m.m.workspace)),
                cast_in=ev_time(lambda: x.half()),
                alloc=ev_time(lambda: torch.empty(M, out_f, device=dev, dtype=torch.half)),
                cast_out=ev_time(lambda: C.to(torch.bfloat16)),
                wrapper=ev_time(lambda: m.m(x.half()).to(torch.bfloat16)))
            for k in tot:
                tot[k] += t[k]
            rows.append((M, name, t))
            print(f"  M {M:4d} {name:9s} ({in_f}->{out_f}): cuBLAS {t['cublas']:.3f}  Marlin kernel {t['kernel']:.3f} "
                  f"({t['kernel']/t['cublas']:.2f}x)  cast-in {t['cast_in']:.3f}  alloc {t['alloc']:.3f}  "
                  f"cast-out {t['cast_out']:.3f}  whole wrapper {t['wrapper']:.3f} ({t['wrapper']/t['cublas']:.2f}x)", flush=True)
        r = tot["kernel"] / tot["cublas"]
        verdict = ("the kernel DELIVERS (<= 0.5x)" if r <= 0.5 else "the kernel does NOT deliver (>= 0.8x)" if r >= 0.8
                   else "between 0.5x and 0.8x")
        print(f"  M {M:4d} per-layer sum: cuBLAS {tot['cublas']:.3f} ms, Marlin kernel {tot['kernel']:.3f} ms -> {r:.3f}x "
              f"[{verdict}]; wrapper {tot['wrapper']:.3f} ms ({tot['wrapper']/tot['cublas']:.3f}x); casts+alloc "
              f"{tot['cast_in'] + tot['alloc'] + tot['cast_out']:.3f} ms", flush=True)
    del mods
    torch.cuda.empty_cache()

    print("\n## (b) one refinement pass under the torch profiler (32 tokens, ctx 700)")
    from torch.profiler import profile, ProfilerActivity
    model = LLaDAModelLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    load_into(model, a.gptq, a.offset)
    torch.manual_seed(0)
    xx = torch.randint(0, 1000, (1, 700), device=dev)
    rp = torch.zeros_like(xx, dtype=torch.bool); rp[:, 600:632] = True

    def one_path(label):
        with torch.no_grad():
            pkv = model(xx, use_cache=True).past_key_values
            for _ in range(20):
                model(xx[:, 600:632], past_key_values=pkv, use_cache=True, replace_position=rp)
            torch.cuda.synchronize()
            walls = []
            for _ in range(50):
                s_, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s_.record(); model(xx[:, 600:632], past_key_values=pkv, use_cache=True, replace_position=rp)
                e_.record(); torch.cuda.synchronize(); walls.append(s_.elapsed_time(e_))
            walls.sort()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                model(xx[:, 600:632], past_key_values=pkv, use_cache=True, replace_position=rp)
                torch.cuda.synchronize()
        ev = prof.key_averages()
        def cuda_t(e):
            return getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) / 1e3
        tot = sum(cuda_t(e) for e in ev)
        gemm = sum(cuda_t(e) for e in ev if any(k in e.key.lower() for k in ("gemm", "marlin", "cutlass", "sgemm", "matmul_kernel", "gemv")))
        launches = sum(e.count for e in ev if e.device_type is not None and "cuda" in str(e.device_type).lower())
        compiled = sorted({e.key for e in ev if any(k in e.key for k in ("Compiled", "compiled", "Torch-Compiled", "CUDAGraph", "cudagraph", "triton"))})
        print(f"### {label}: pass wall median {walls[len(walls)//2]:.2f} ms (CUDA events, 50 passes); profiled CUDA self time "
              f"{tot:.2f} ms, of which GEMM kernels {gemm:.2f} ms ({100*gemm/max(tot,1e-9):.1f} %), everything else "
              f"{tot-gemm:.2f} ms; CUDA kernel launches {launches}")
        print(f"    compiled / graphed regions seen: {compiled if compiled else 'none'}")
        top = sorted(ev, key=lambda e: -cuda_t(e))[:14]
        for e in top:
            print(f"    {cuda_t(e):8.3f} ms  x{e.count:<5d} {e.key[:110]}")
        return walls[len(walls)//2], gemm, tot
    wb, gb, tb = one_path("bf16 (expanded weights)")
    swap_kernel(model, a.gptq, a.offset, "marlin")
    wm, gm, tm = one_path("Marlin (block-start pass on bf16 copy)")
    print(f"\nsummary (b): pass wall Marlin/bf16 {wm/wb:.3f}; GEMM time Marlin/bf16 {gm/max(gb,1e-9):.3f}; "
          f"non-GEMM CUDA time Marlin - bf16 {(tm-gm)-(tb-gb):+.2f} ms")


if __name__ == "__main__":
    main()
