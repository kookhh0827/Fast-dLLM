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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check"])
    ap.add_argument("--gptq", required=True)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--layers", default="0,15,31")
    a = ap.parse_args()
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


if __name__ == "__main__":
    main()
