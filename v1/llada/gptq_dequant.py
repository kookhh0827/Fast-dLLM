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
    ap.add_argument("cmd", choices=["check", "kernelcheck", "marlincheck"])
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


if __name__ == "__main__":
    main()
