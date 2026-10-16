"""Phase 3 stage 3-0: the A compressions of LLaDA-8B-Instruct that Phase 3 adds (results/phase3/PREREG.md, section 3.1
and gate C of section 2.3; docs/plan/07_phase3.md, section 3.2). The other A compressions are older: the depth cut
(KL-greedy set), GPTQ 4-bit (`gptq_dequant.py`, Marlin) and SoloQ W4A4 (imported from outside the fork).

- fp8: FP8 E4M3 weights (one scale per output channel) and activations (one scale per token, computed at run time).
  Real kernel: torch._scaled_mm with row-wise scales, on every pass. No calibration.
- awq4: AWQ 4-bit weights, group 128, asymmetric (zero points), as llm-awq (github.com/mit-han-lab/llm-awq): the
  scale search (20 grid points) on the groups (attn_norm -> q, k, v), (v_proj -> attn_out), (ff_norm -> ff_proj,
  up_proj) and (up_proj -> ff_out), then the clip search (20 grid points, shrink up to 0.5; not on q and k). The
  calibration data: PileVal, 128 documents of at most 512 tokens, joined and cut into blocks of 512 (seed 42).
  Real kernel: torch's tinygemm int4 kernel (aten._weight_int4pack_mm) on calls with at most 64 rows (the refinement
  passes); a bf16 copy of the expanded weights on larger calls (the cache-writing passes), as Marlin in Phase 2.
- quarot_w4a4: QuaRot (github.com/spcl/QuaRot, the command of its README: --rotate --a_bits 4 --w_bits 4 --w_clip):
  the norms are folded into the next linears; a random Hadamard rotation of the residual stream; online Hadamard
  transforms before attn_out (4096) and ff_out (12288 = 12 x 1024); weights 4-bit, per output channel, symmetric,
  GPTQ with the clip search (WikiText-2, 128 x 2048, damp 0.01, no act-order, true-sequential groups); activations
  4-bit, per token, symmetric, clip ratio 1.0. The embedding and the LM head stay in bf16. Simulated (fake
  quantization), so its speed is a projection.
- sparse24: 2:4 sparsity with SparseGPT (github.com/IST-DASLab/sparsegpt, prunen 2, prunem 4, block 128, damp
  0.01; C4 train shard 0, 128 x 2048, seed 0). Real kernel: the semi-structured sparse tensor of PyTorch
  (torch.sparse.to_sparse_semi_structured) on every pass.

A calibrated compression is made once (`make`, in a GPU job) and saved to a folder; a cell loads it (`apply`).
Gate C1 (`c1_check`) compares each compressed linear with a reference implementation on 100 random inputs.
"""
import glob
import gzip
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

LINEARS = ["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"]
METHODS = ("fp8", "awq4", "quarot_w4a4", "sparse24")
FP8_MAX = 448.0
HUB = "/project2/ppanda_1750/kook/dllm/models/hub"


def blocks(model):
    return model.model.transformer.blocks


def _snap(repo):
    s = sorted(glob.glob(f"{HUB}/datasets--{repo.replace('/', '--')}/snapshots/*"))
    assert s, f"dataset {repo} is not in the cache"
    return s[0]


# ----------------------------------------------------------------------------------------------- calibration data
def calib_wikitext2(tok, n=128, seqlen=2048, seed=0):
    """QuaRot (data_utils.get_wikitext2): the train split joined with blank lines, n random windows."""
    import pandas as pd
    df = pd.read_parquet(f"{_snap('Salesforce/wikitext')}/wikitext-2-raw-v1/train-00000-of-00001.parquet")
    enc = tok("\n\n".join(df["text"].tolist()), return_tensors="pt").input_ids[0]
    random.seed(seed)
    out = []
    for _ in range(n):
        i = random.randint(0, enc.numel() - seqlen - 1)
        out.append(enc[i:i + seqlen])
    return torch.stack(out)


def calib_c4(tok, n=128, seqlen=2048, seed=0):
    """SparseGPT (datautils.get_c4): random documents of the train shard 0 with more than seqlen tokens."""
    path = f"{_snap('allenai/c4')}/en/c4-train.00000-of-01024.json.gz"
    docs = [json.loads(line)["text"] for line in gzip.open(path, "rt")]
    random.seed(seed)
    out = []
    while len(out) < n:
        enc = tok(docs[random.randint(0, len(docs) - 1)], return_tensors="pt").input_ids[0]
        if enc.numel() <= seqlen:
            continue
        j = random.randint(0, enc.numel() - seqlen - 1)
        out.append(enc[j:j + seqlen])
    return torch.stack(out)


def calib_pileval(tok, n=128, block=512, seed=42):
    """llm-awq (calib_data.get_calib_dataset): the validation set shuffled with seed 42; the first n documents of at
    most `block` tokens, joined and cut into blocks of `block` tokens."""
    from datasets import load_dataset
    ds = load_dataset("json", data_files=f"{_snap('mit-han-lab/pile-val-backup')}/val.jsonl.zst",
                      split="train").shuffle(seed=seed)
    samples = []
    for d in ds:
        enc = tok.encode(d["text"].strip())
        if len(enc) > block or len(enc) == 0:
            continue
        samples.append(torch.tensor(enc))
        if len(samples) == n:
            break
    cat = torch.cat(samples)
    k = cat.numel() // block
    return cat[:k * block].view(k, block)


# ------------------------------------------------------------------------------------------- the layer loop helpers
@torch.no_grad()
def embed(model, ids, dev):
    return model.model.transformer.wte(ids.to(dev))


@torch.no_grad()
def run_block(blk, X, bs=4):
    out = torch.empty_like(X)
    for i in range(0, X.shape[0], bs):
        out[i:i + bs] = blk(X[i:i + bs])[0]
    return out


class InputCapture:
    """Forward pre-hooks that collect the input of named linears: a Hessian sum (GPTQ, SparseGPT) or the raw rows."""

    def __init__(self, mods, mode="hessian", max_rows=None):
        self.mode, self.max_rows, self.store, self.hooks = mode, max_rows, {}, []
        for name, m in mods.items():
            self.hooks.append(m.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name):
        def f(mod, args):
            x = args[0].reshape(-1, args[0].shape[-1])
            if self.mode == "hessian":
                xf = x.float()
                H, n = self.store.get(name, (None, 0))
                h = xf.t() @ xf
                self.store[name] = (h if H is None else H + h, n + xf.shape[0])
            else:
                self.store.setdefault(name, []).append(x.detach())
        return f

    def remove(self):
        for h in self.hooks:
            h.remove()

    def hessian(self, name):
        H, n = self.store[name]
        return H / n

    def rows(self, name):
        return torch.cat(self.store[name])


def _hinv(H, percdamp):
    """The upper Cholesky factor of the damped inverse Hessian (GPTQ, SparseGPT); dead columns get 1 on the diagonal."""
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(H.shape[0], device=H.device)
    L = torch.linalg.cholesky(H)
    return torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True), dead


# --------------------------------------------------------------------------------------------------------- quantizers
def sym_scale_perchannel(W, bits, clip_search=False, grid=100, maxshrink=0.8, norm=2.4):
    """QuaRot's WeightQuantizer (symmetric, per output channel; levels -(2^(b-1)) .. 2^(b-1)-1, scale = max/maxq),
    with the MSE clip search of --w_clip (norm 2.4, grid 100, shrink up to 0.8)."""
    maxq = 2 ** (bits - 1) - 1
    Wf = W.float()
    xmax = Wf.abs().amax(1).clamp(min=1e-8)
    scale = xmax / maxq
    if clip_search:
        best = torch.full_like(xmax, float("inf"))
        for i in range(int(maxshrink * grid)):
            p = 1 - i / grid
            s1 = (p * xmax) / maxq
            q = torch.clamp(torch.round(Wf / s1[:, None]), -(maxq + 1), maxq) * s1[:, None]
            err = (q - Wf).abs().pow(norm).sum(1)
            better = err < best
            best = torch.where(better, err, best)
            scale = torch.where(better, s1, scale)
    return scale


def sym_quant(w, scale, bits):
    maxq = 2 ** (bits - 1) - 1
    return torch.clamp(torch.round(w / scale), -(maxq + 1), maxq) * scale


def act_quant_sym(x, bits=4, clip=1.0):
    """QuaRot's ActQuantizer: symmetric, one scale per token (row), levels -(2^(b-1)) .. 2^(b-1)-1."""
    maxq = 2 ** (bits - 1) - 1
    xf = x.float()
    s = (xf.abs().amax(-1, keepdim=True) * clip / maxq).clamp(min=1e-8)
    return (torch.clamp(torch.round(xf / s), -(maxq + 1), maxq) * s).to(x.dtype)


def asym_quant_groups(W, bits=4, group=128):
    """llm-awq pseudo_quantize_tensor (zero_point=True): per (row, group) min/max, integer zero point.
    Returns the dequantized weight, the codes (rows x cols, 0..15), the scales and the zeros (rows x groups)."""
    out_f, in_f = W.shape
    G = W.float().reshape(out_f, in_f // group, group)
    mx, mn = G.amax(-1, keepdim=True), G.amin(-1, keepdim=True)
    maxq = 2 ** bits - 1
    s = ((mx - mn).clamp(min=1e-5) / maxq)
    z = torch.clamp(-torch.round(mn / s), 0, maxq)
    q = torch.clamp(torch.round(G / s) + z, 0, maxq)
    deq = ((q - z) * s).reshape(out_f, in_f)
    return deq, q.reshape(out_f, in_f).to(torch.uint8), s.squeeze(-1), z.squeeze(-1).to(torch.uint8)


def gptq_sym_perchannel(W, H, bits=4, percdamp=0.01, blocksize=128, clip_search=True):
    """GPTQ (github.com/IST-DASLab/gptq, fasterquant) with QuaRot's static per-channel symmetric scales."""
    W = W.float().clone()
    scale = sym_scale_perchannel(W, bits, clip_search)[:, None]
    Hinv, dead = _hinv(H, percdamp)
    W[:, dead] = 0
    Q = torch.zeros_like(W)
    ncol = W.shape[1]
    for i1 in range(0, ncol, blocksize):
        i2 = min(i1 + blocksize, ncol)
        W1 = W[:, i1:i2].clone()
        Q1, E1 = torch.zeros_like(W1), torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            q = sym_quant(w[:, None], scale, bits).flatten()
            Q1[:, i] = q
            e = (w - q) / Hi[i, i]
            W1[:, i:] -= e[:, None] @ Hi[i, i:][None, :]
            E1[:, i] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


def sparsegpt_24(W, H, percdamp=0.01, blocksize=128, prunen=2, prunem=4):
    """SparseGPT (github.com/IST-DASLab/sparsegpt, fasterprune) with the n:m mask, no quantization."""
    W = W.float().clone()
    Hinv, dead = _hinv(H, percdamp)
    W[:, dead] = 0
    ncol = W.shape[1]
    for i1 in range(0, ncol, blocksize):
        i2 = min(i1 + blocksize, ncol)
        W1 = W[:, i1:i2].clone()
        Q1, E1 = torch.zeros_like(W1), torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        mask = torch.zeros_like(W1, dtype=torch.bool)
        for i in range(i2 - i1):
            w = W1[:, i]
            if i % prunem == 0:
                tmp = W1[:, i:i + prunem] ** 2 / (torch.diag(Hi)[i:i + prunem].reshape(1, -1)) ** 2
                mask.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
            q = w.clone()
            q[mask[:, i]] = 0
            Q1[:, i] = q
            e = (w - q) / Hi[i, i]
            W1[:, i:] -= e[:, None] @ Hi[i, i:][None, :]
            E1[:, i] = e
        W[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return W


# ----------------------------------------------------------------------------------------------------- Hadamard
def _sylvester(n):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    assert H.shape[0] == n, n
    return H


def _paley12():
    """A Hadamard matrix of order 12 (Paley construction I, q = 11)."""
    q = 11
    res = {(i * i) % q for i in range(1, q)}
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in res else -1)                        # noqa: E731
    Q = torch.tensor([[chi(j - i) for j in range(q)] for i in range(q)], dtype=torch.float64)
    S = torch.zeros(q + 1, q + 1, dtype=torch.float64)
    S[0, 1:], S[1:, 0], S[1:, 1:] = 1, -1, Q
    H = S + torch.eye(q + 1, dtype=torch.float64)
    assert torch.allclose(H @ H.t(), (q + 1) * torch.eye(q + 1, dtype=torch.float64))
    return H


def hadamard_factors(n):
    """n = a * m with m a power of two and a in {1, 12}: the two normalized factors (H_a, H_m)."""
    for a, Ha in ((12, None), (1, None)):
        if n % a == 0 and (n // a) & (n // a - 1) == 0:
            m = n // a
            HA = _paley12() if a == 12 else torch.ones(1, 1, dtype=torch.float64)
            return HA / math.sqrt(a), _sylvester(m) / math.sqrt(m)
    raise ValueError(f"no Hadamard construction for size {n}")


class Hadamard:
    """x -> x @ (H_a kron H_m), applied as H_a^T X H_m on x reshaped to (a, m)."""

    def __init__(self, n, dev):
        Ha, Hm = hadamard_factors(n)
        self.n, self.a, self.m = n, Ha.shape[0], Hm.shape[0]
        self.Ha, self.Hm = Ha.float().to(dev), Hm.float().to(dev)

    def __call__(self, x):
        shp = x.shape
        X = x.float().reshape(-1, self.a, self.m) @ self.Hm
        if self.a > 1:
            X = torch.einsum("ik,bim->bkm", self.Ha, X)
        return X.reshape(shp).to(x.dtype)

    def matrix(self):
        return torch.kron(self.Ha.double(), self.Hm.double())


# ----------------------------------------------------------------------------------------------- compressed linears
class FP8Linear(torch.nn.Module):
    """FP8 E4M3 weights (one scale per output channel) and activations (one scale per token, at run time);
    torch._scaled_mm with row-wise scales, bf16 output. Rows are padded to a multiple of 16."""

    kernel = "torch._scaled_mm (FP8 E4M3, row-wise scales)"

    def __init__(self, W):
        super().__init__()
        self.out_features, self.in_features = W.shape
        s = (W.float().abs().amax(1, keepdim=True) / FP8_MAX).clamp(min=1e-12)          # (out, 1)
        wq = (W.float() / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        self.register_buffer("wq", wq)                                                   # (out, in), row-major
        self.register_buffer("ws", s.t().contiguous())                                   # (1, out)

    def quant_x(self, x2):
        xs = (x2.float().abs().amax(1, keepdim=True) / FP8_MAX).clamp(min=1e-12)          # (M, 1)
        return (x2.float() / xs).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), xs

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features)
        M = x2.shape[0]
        pad = (-M) % 16
        if pad:
            x2 = F.pad(x2, (0, 0, 0, pad))
        xq, xs = self.quant_x(x2)
        y = torch._scaled_mm(xq, self.wq.t(), scale_a=xs, scale_b=self.ws, out_dtype=torch.bfloat16)
        return y[:M].reshape(*shp[:-1], self.out_features)

    def reference(self, x):
        """The fake-quantization reference (fp32): the same FP8 values, dequantized, and a fp32 matmul."""
        x2 = x.reshape(-1, self.in_features)
        xq, xs = self.quant_x(x2)
        return (xq.float() * xs) @ (self.wq.float() * self.ws.t()).t()


class Sparse24Linear(torch.nn.Module):
    """2:4 sparse weights in PyTorch's semi-structured sparse format; F.linear runs the sparse kernel."""

    kernel = "torch semi-structured sparse (2:4)"

    def __init__(self, W):
        super().__init__()
        from torch.sparse import to_sparse_semi_structured
        self.out_features, self.in_features = W.shape
        self.register_buffer("w_dense", W)                                               # for C1 and the record
        self.w = to_sparse_semi_structured(W)

    def forward(self, x):
        shp = x.shape
        return F.linear(x.reshape(-1, self.in_features), self.w).reshape(*shp[:-1], self.out_features)

    def reference(self, x):
        return x.reshape(-1, self.in_features).float() @ self.w_dense.float().t()


class QuaRotLinear(torch.nn.Module):
    """QuaRot fake quantization: (online Hadamard), 4-bit per-token symmetric activations, 4-bit weights (stored
    dequantized), bf16 matmul."""

    kernel = "none (simulated)"

    def __init__(self, Wq, had=None, a_bits=4, a_clip=1.0):
        super().__init__()
        self.out_features, self.in_features = Wq.shape
        self.register_buffer("w", Wq)
        self.had, self.a_bits, self.a_clip = had, a_bits, a_clip
        self.quant_act = True           # off while GPTQ collects its Hessians (QuaRot quantizes activations later)
        self.capture = None             # GPTQ: a function that receives the input after the online Hadamard

    def forward(self, x):
        if self.had is not None:
            x = self.had(x)
        if self.capture is not None:
            self.capture(x)
        if self.quant_act:
            x = act_quant_sym(x, self.a_bits, self.a_clip)
        return F.linear(x, self.w)

    def reference(self, x):
        """fp64: the Hadamard as one dense matrix, then the same quantizer formula, then the matmul."""
        x2 = x.reshape(-1, self.in_features).double()
        if self.had is not None:
            x2 = x2 @ self.had.matrix().to(x2.device)
        maxq = 2 ** (self.a_bits - 1) - 1
        s = (x2.abs().amax(-1, keepdim=True) * self.a_clip / maxq).clamp(min=1e-8)
        xq = torch.clamp(torch.round(x2 / s), -(maxq + 1), maxq) * s
        return xq.to(torch.bfloat16).double() @ self.w.double().t()


def awq_linear(codes, scales, zeros, dtype, row_max=64):
    """AWQ weights on torch's tinygemm int4 kernel (`gptq_dequant.Int4PackLinear`), with a bf16 copy of the
    expanded weights for calls with more than row_max rows (`gptq_dequant.Int4DispatchLinear`)."""
    import gptq_dequant as gq
    out_f, in_f = codes.shape
    deq = ((codes.float().reshape(out_f, in_f // 128, 128) - zeros.float()[..., None]) * scales.float()[..., None])
    deq = deq.reshape(out_f, in_f).to(dtype)
    k = gq.Int4PackLinear(codes.int(), scales.float().t().contiguous(), zeros.float().t().contiguous(), out_f, in_f,
                          dtype)
    m = gq.Int4DispatchLinear(deq, k, row_max)
    m.kernel = f"aten._weight_int4pack_mm (tinygemm int4, group 128) on <= {row_max} rows; bf16 above"
    return m


# ------------------------------------------------------------------------------------------------------- AWQ
@torch.no_grad()
def _awq_search_scale(x, linears, inspect, org_out, n_grid=20):
    """llm-awq auto_scale._search_module_scale: s = mean|x|^alpha (normalized), alpha on a grid of 20; the weights
    are fake-quantized (4-bit, group 128, asymmetric); the scale with the smallest MSE of the inspected output."""
    x_max = x.abs().reshape(-1, x.shape[-1]).float().mean(0)
    org = {id(l): l.weight.data.clone() for l in linears}
    best, best_s = float("inf"), None
    for r in range(n_grid):
        a = r / n_grid
        s = x_max.pow(a).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        for l in linears:
            l.weight.data = (asym_quant_groups(org[id(l)].float() * s[None, :])[0] / s[None, :]).to(org[id(l)].dtype)
        out = inspect()
        loss = (org_out - out).float().pow(2).mean().item()
        if loss < best:
            best, best_s = loss, s.clone()
    for l in linears:
        l.weight.data = org[id(l)]
    return best_s, best


@torch.no_grad()
def _awq_clip(lin, x, n_grid=20, max_shrink=0.5, n_tokens=512, oc_batch=256, group=128):
    """llm-awq auto_clip_layer: per (output channel, group), the max |w| shrink with the smallest output MSE on
    `n_tokens` sampled input rows."""
    W = lin.weight.data.float()
    out_f, in_f = W.shape
    xs = x.reshape(-1, in_f)
    xs = xs[torch.linspace(0, xs.shape[0] - 1, n_tokens).long()].float()                 # (T, in), as llm-awq
    xg = xs.reshape(1, n_tokens, in_f // group, group)                                     # (1, T, G, g)
    best_max = torch.empty(out_f, in_f // group, 1, device=W.device)
    for o in range(0, out_f, oc_batch):
        w = W[o:o + oc_batch].reshape(-1, 1, in_f // group, group)                         # (oc, 1, G, g)
        org_max = w.abs().amax(-1, keepdim=True)                                           # (oc, 1, G, 1)
        bmax = org_max.clone()
        min_err = torch.full_like(org_max, float("inf"))
        org_out = (xg * w).sum(-1)                                                         # (oc, T, G)
        for i in range(int(max_shrink * n_grid)):
            mx = org_max * (1 - i / n_grid)
            cw = torch.clamp(w, -mx, mx)
            q = asym_quant_groups(cw.reshape(-1, group).reshape(cw.shape[0], -1))[0].reshape(cw.shape)
            err = ((xg * q).sum(-1) - org_out).pow(2).mean(1).view(min_err.shape)
            better = err < min_err
            min_err[better] = err[better]
            bmax[better] = mx[better]
        best_max[o:o + oc_batch] = bmax.squeeze(1)
    Wg = W.reshape(out_f, in_f // group, group)
    lin.weight.data = torch.clamp(Wg, -best_max, best_max).reshape(out_f, in_f).to(lin.weight.dtype)


# ------------------------------------------------------------------------------------------------------- AWQ
@torch.no_grad()
def _awq_search(x, linears, inspect, n_grid=20, chunk=4):
    """llm-awq auto_scale._search_module_scale: s = mean|x|^alpha, normalized by sqrt(max * min), alpha = r / 20;
    the weights are fake-quantized (4-bit, group 128, asymmetric); the smallest MSE of the inspected output wins.
    The weights are restored after each trial."""
    x_max = x.abs().reshape(-1, x.shape[-1]).float().mean(0)
    org_w = [l.weight.data.clone() for l in linears]
    org_out = inspect(x)
    best, best_s, hist = float("inf"), None, []
    for r in range(n_grid):
        s = x_max.pow(r / n_grid).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        for l, w in zip(linears, org_w):
            l.weight.data = (asym_quant_groups(w.float() * s[None, :])[0] / s[None, :]).to(w.dtype)
        loss = (org_out.float() - inspect(x).float()).pow(2).mean().item()
        hist.append(loss)
        if loss < best:
            best, best_s = loss, s.clone()
    for l, w in zip(linears, org_w):
        l.weight.data = w
    return best_s, best, hist.index(best) / n_grid


@torch.no_grad()
def _awq_clip(lin, x, n_grid=20, max_shrink=0.5, n_tokens=512, group=128):
    """llm-awq auto_clip_layer: per (output channel, group), the max |w| (shrunk by up to 0.5) with the smallest
    output MSE on about 512 sampled input tokens; then the weights are clamped to it."""
    W = lin.weight.data.float()
    out_f, in_f = W.shape
    xs = x.reshape(-1, in_f).float()
    xs = xs[0::max(1, xs.shape[0] // n_tokens)]
    xg = xs.reshape(1, xs.shape[0], in_f // group, group)
    oc = 256 if out_f % 256 == 0 else 64
    best_max = []
    for o in range(0, out_f, oc):
        w = W[o:o + oc].reshape(-1, 1, in_f // group, group)
        org_max = w.abs().amax(-1, keepdim=True)
        bmax, min_err = org_max.clone(), torch.full_like(org_max, 1e9)
        org_out = (xg * w).sum(-1)
        for i in range(int(max_shrink * n_grid)):
            mx = org_max * (1 - i / n_grid)
            cw = torch.clamp(w, -mx, mx)
            q = asym_quant_groups(cw.reshape(cw.shape[0], -1))[0].reshape(cw.shape)
            err = ((xg * q).sum(-1) - org_out).pow(2).mean(1).view(min_err.shape)
            better = err < min_err
            min_err[better], bmax[better] = err[better], mx[better]
        best_max.append(bmax.squeeze(1))
    bm = torch.cat(best_max)                                                              # (out, G, 1)
    lin.weight.data = torch.clamp(W.reshape(out_f, in_f // group, group), -bm, bm).reshape(out_f, in_f).to(
        lin.weight.dtype)


@torch.no_grad()
def make_awq(model, tok, dev, log=print):
    """llm-awq run_awq on LLaDA: for each block, the input features of the linears and the next block's inputs come
    from the unchanged block; the four scale searches run on the unchanged weights; then the scales are applied (and
    the stored features divided by them); then the clip search (not on q and k)."""
    ids = calib_pileval(tok)
    log(f"[awq] calibration: PileVal, {tuple(ids.shape)} tokens")
    X = embed(model, ids, dev)
    for l, blk in enumerate(blocks(model)):
        t0 = time.time()
        cap = InputCapture({n: getattr(blk, n) for n in ("q_proj", "attn_out", "ff_proj", "ff_out")}, mode="rows")
        Y = run_block(blk, X)
        cap.remove()
        f = {n: cap.rows(n).reshape(X.shape[0], X.shape[1], -1) for n in ("q_proj", "attn_out", "ff_proj", "ff_out")}
        del cap

        def attn(xn):
            o = torch.empty(*xn.shape[:2], blk.attn_out.out_features, device=dev, dtype=xn.dtype)
            for i in range(0, xn.shape[0], 4):
                c = xn[i:i + 4]
                o[i:i + 4] = blk.attention(blk.q_proj(c), blk.k_proj(c), blk.v_proj(c), None)[0]
            return o

        def mlp(xn):
            return blk.ff_out(blk.act(blk.ff_proj(xn)) * blk.up_proj(xn))

        s1, l1, a1 = _awq_search(f["q_proj"], [blk.q_proj, blk.k_proj, blk.v_proj], attn)
        s2, l2, a2 = _awq_search(f["attn_out"], [blk.attn_out], blk.attn_out)
        s3, l3, a3 = _awq_search(f["ff_proj"], [blk.ff_proj, blk.up_proj], mlp)
        s4, l4, a4 = _awq_search(f["ff_out"], [blk.ff_out], blk.ff_out)

        def scale_cols(lin, s):
            lin.weight.data = (lin.weight.float() * s[None, :]).to(lin.weight.dtype)

        def scale_rows(lin, s):
            lin.weight.data = (lin.weight.float() / s[:, None]).to(lin.weight.dtype)

        blk.attn_norm.weight.data = (blk.attn_norm.weight.float() / s1).to(blk.attn_norm.weight.dtype)
        for lin in (blk.q_proj, blk.k_proj, blk.v_proj):
            scale_cols(lin, s1)
        scale_rows(blk.v_proj, s2)
        scale_cols(blk.attn_out, s2)
        blk.ff_norm.weight.data = (blk.ff_norm.weight.float() / s3).to(blk.ff_norm.weight.dtype)
        for lin in (blk.ff_proj, blk.up_proj):
            scale_cols(lin, s3)
        scale_rows(blk.up_proj, s4)
        scale_cols(blk.ff_out, s4)
        xin = {"v_proj": f["q_proj"].float() / s1, "attn_out": f["attn_out"].float() / s2,
               "ff_proj": f["ff_proj"].float() / s3, "up_proj": f["ff_proj"].float() / s3,
               "ff_out": f["ff_out"].float() / s4}
        del f
        for n, xi in xin.items():
            _awq_clip(getattr(blk, n), xi)
        del xin
        X = Y
        torch.cuda.empty_cache()
        log(f"[awq] block {l}: alpha qkv {a1:.2f} v-o {a2:.2f} mlp {a3:.2f} up-down {a4:.2f}; "
            f"mse {l1:.2e} {l2:.2e} {l3:.2e} {l4:.2e} ({time.time() - t0:.0f} s)")
    art = {}
    for l, blk in enumerate(blocks(model)):
        for n in LINEARS:
            _, q, s, z = asym_quant_groups(getattr(blk, n).weight.data)
            art[f"{l}.{n}.codes"], art[f"{l}.{n}.scales"], art[f"{l}.{n}.zeros"] = q.cpu(), s.float().cpu(), z.cpu()
        art[f"{l}.attn_norm"] = blk.attn_norm.weight.data.cpu()
        art[f"{l}.ff_norm"] = blk.ff_norm.weight.data.cpu()
    return art, dict(calibration="mit-han-lab/pile-val-backup val.jsonl.zst, shuffle seed 42, 128 documents of at "
                                 "most 512 tokens, joined, blocks of 512", calib_shape=list(ids.shape),
                     bits=4, group=128, zero_point=True, scale_grid=20, clip_grid=20, clip_max_shrink=0.5)


# --------------------------------------------------------------------------------------------------------- QuaRot
@torch.no_grad()
def quarot_rotate(model, seed=0):
    """Fold the RMS norm weights into the next linears; rotate the residual stream by Q = diag(random signs) H
    (QuaRot random_hadamard_matrix, seed); fold the online Hadamards of attn_out (d) and ff_out (mlp width) into
    their weights (W <- W H). The model function does not change (checked in `make_quarot`)."""
    tr = model.model.transformer
    dev, dt = tr.wte.weight.device, tr.wte.weight.dtype
    d = tr.wte.weight.shape[1]

    def fold(norm, lins):
        g = norm.weight.data.double()
        for lin in lins:
            lin.weight.data = (lin.weight.data.double() * g[None, :]).to(dt)
        norm.weight.data.fill_(1.0)

    for blk in tr.blocks:
        fold(blk.attn_norm, [blk.q_proj, blk.k_proj, blk.v_proj])
        fold(blk.ff_norm, [blk.ff_proj, blk.up_proj])
    fold(tr.ln_f, [tr.ff_out])
    gen = torch.Generator().manual_seed(seed)
    sgn = torch.randint(0, 2, (d,), generator=gen).double() * 2 - 1
    Q = (sgn[:, None] * Hadamard(d, "cpu").matrix()).to(dev)                                # diag(s) @ H
    tr.wte.weight.data = (tr.wte.weight.data.double() @ Q).to(dt)
    for blk in tr.blocks:
        for lin in (blk.q_proj, blk.k_proj, blk.v_proj, blk.ff_proj, blk.up_proj):
            lin.weight.data = (lin.weight.data.double() @ Q).to(dt)
        for lin in (blk.attn_out, blk.ff_out):
            lin.weight.data = (Q.t() @ lin.weight.data.double()).to(dt)
    tr.ff_out.weight.data = (tr.ff_out.weight.data.double() @ Q).to(dt)
    had_o, had_d = Hadamard(d, dev), Hadamard(tr.blocks[0].ff_out.in_features, dev)
    for blk in tr.blocks:
        blk.attn_out.weight.data = had_o(blk.attn_out.weight.data.double()).to(dt)
        blk.ff_out.weight.data = had_d(blk.ff_out.weight.data.double()).to(dt)
    return had_o, had_d


def quarot_wrap(model, had_o, had_d):
    """Replace each block linear by a QuaRotLinear (online Hadamard before attn_out and ff_out)."""
    for blk in blocks(model):
        for n in LINEARS:
            had = had_o if n == "attn_out" else (had_d if n == "ff_out" else None)
            setattr(blk, n, QuaRotLinear(getattr(blk, n).weight.data, had))


class _QCapture:
    """Hessians of the inputs of QuaRotLinear modules after their online Hadamard, with activation quantization off."""

    def __init__(self, mods):
        self.mods, self.store = mods, {}
        for name, m in mods.items():
            m.capture = self._fn(name)

    def _fn(self, name):
        def f(x):
            xf = x.reshape(-1, x.shape[-1]).float()
            H, n = self.store.get(name, (None, 0))
            h = xf.t() @ xf
            self.store[name] = (h if H is None else H + h, n + xf.shape[0])
        return f

    def remove(self):
        for m in self.mods.values():
            m.capture = None

    def hessian(self, name):
        H, n = self.store[name]
        return H / n


@torch.no_grad()
def _logits(model, ids):
    return model(ids).logits.float()


@torch.no_grad()
def make_quarot(model, tok, dev, log=print, seed=0):
    ids = calib_wikitext2(tok)
    log(f"[quarot] calibration: WikiText-2 train, {tuple(ids.shape)} tokens")
    probe = ids[:2, :256].to(dev)
    ref = _logits(model, probe)
    had_o, had_d = quarot_rotate(model, seed)
    quarot_wrap(model, had_o, had_d)
    for blk in blocks(model):
        for n in LINEARS:
            getattr(blk, n).quant_act = False
    # the rotated model with its online Hadamards and no quantization must compute the same function
    rot = _logits(model, probe)
    inv = float((rot - ref).norm() / ref.norm())
    log(f"[quarot] rotation check: relative logit difference {inv:.3e} (no quantization, 2 x 256 tokens)")
    X = embed(model, ids, dev)
    groups = [["q_proj", "k_proj", "v_proj"], ["attn_out"], ["ff_proj", "up_proj"], ["ff_out"]]
    for l, blk in enumerate(blocks(model)):
        t0 = time.time()
        for g in groups:
            cap = _QCapture({n: getattr(blk, n) for n in g})
            run_block(blk, X)
            cap.remove()
            for n in g:
                m = getattr(blk, n)
                m.w.data = gptq_sym_perchannel(m.w.data, cap.hessian(n)).to(m.w.dtype)
            del cap
        X = run_block(blk, X)
        torch.cuda.empty_cache()
        log(f"[quarot] block {l}: GPTQ done ({time.time() - t0:.0f} s)")
    for blk in blocks(model):
        for n in LINEARS:
            getattr(blk, n).quant_act = True
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return sd, dict(calibration="Salesforce/wikitext wikitext-2-raw-v1 train, joined with blank lines, 128 random "
                                "windows of 2048 tokens, seed 0", rotation_seed=seed, rotation_check_rel_logit=inv,
                    w_bits=4, w_scheme="per output channel, symmetric, GPTQ (damp 0.01, block 128, no act-order, "
                                       "true-sequential groups), MSE clip search (norm 2.4, grid 100, shrink 0.8)",
                    a_bits=4, a_scheme="per token, symmetric, clip 1.0", online_hadamard=dict(
                        attn_out=[had_o.a, had_o.m], ff_out=[had_d.a, had_d.m]))


# ------------------------------------------------------------------------------------------------------ SparseGPT
@torch.no_grad()
def make_sparse24(model, tok, dev, log=print):
    ids = calib_c4(tok)
    log(f"[sparsegpt] calibration: C4 train shard 0, {tuple(ids.shape)} tokens")
    X = embed(model, ids, dev)
    for l, blk in enumerate(blocks(model)):
        t0 = time.time()
        cap = InputCapture({n: getattr(blk, n) for n in LINEARS})
        run_block(blk, X)
        cap.remove()
        for n in LINEARS:
            lin = getattr(blk, n)
            lin.weight.data = sparsegpt_24(lin.weight.data, cap.hessian(n)).to(lin.weight.dtype)
        del cap
        X = run_block(blk, X)
        torch.cuda.empty_cache()
        log(f"[sparsegpt] block {l}: pruned ({time.time() - t0:.0f} s)")
    art = {f"{l}.{n}": getattr(blk, n).weight.data.cpu() for l, blk in enumerate(blocks(model)) for n in LINEARS}
    return art, dict(calibration="allenai/c4 en/c4-train.00000-of-01024.json.gz, 128 random windows of 2048 tokens "
                                 "from documents longer than 2048 tokens, seed 0", calib_shape=list(ids.shape),
                     prunen=2, prunem=4, blocksize=128, percdamp=0.01)


# ------------------------------------------------------------------------------------------------- save and apply
def save(art, meta, out_dir, method):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    save_file({k: v.contiguous() for k, v in art.items()}, os.path.join(out_dir, f"{method}.safetensors"))
    json.dump(meta, open(os.path.join(out_dir, f"{method}.json"), "w"), indent=1)


@torch.no_grad()
def apply(model, method, art_dir=None, row_max=64):
    """Put a compression into a loaded bf16 model. Returns the record of the compression for summary.json.
    `row_max` (awq4): calls with more rows use the bf16 copy (the runner passes --marlin-row-max)."""
    from safetensors.torch import load_file
    dev, dt = model.model.transformer.wte.weight.device, model.model.transformer.wte.weight.dtype
    rec = dict(kind=None, method=method, artifact=None, kernel=None)
    if method == "fp8":
        for blk in blocks(model):
            for n in LINEARS:
                setattr(blk, n, FP8Linear(getattr(blk, n).weight.data))
        rec.update(kind="weights+activations", bits=8, kernel=FP8Linear.kernel,
                   scheme="E4M3; weights: one scale per output channel; activations: one scale per token at run time")
    else:
        path = os.path.join(art_dir, f"{method}.safetensors")
        art = load_file(path, device=str(dev))
        meta = json.load(open(os.path.join(art_dir, f"{method}.json")))
        rec.update(artifact=path, meta=meta)
        if method == "awq4":
            for l, blk in enumerate(blocks(model)):
                blk.attn_norm.weight.data.copy_(art[f"{l}.attn_norm"])
                blk.ff_norm.weight.data.copy_(art[f"{l}.ff_norm"])
                for n in LINEARS:
                    setattr(blk, n, awq_linear(art[f"{l}.{n}.codes"], art[f"{l}.{n}.scales"], art[f"{l}.{n}.zeros"],
                                               dt, row_max))
            rec.update(kind="weights", bits=4, kernel=blocks(model)[0].q_proj.kernel, row_max=row_max)
        elif method == "quarot_w4a4":
            hd = meta["online_hadamard"]
            had_o, had_d = Hadamard(hd["attn_out"][0] * hd["attn_out"][1], dev), Hadamard(
                hd["ff_out"][0] * hd["ff_out"][1], dev)
            quarot_wrap(model, had_o, had_d)
            model.load_state_dict(art, strict=True)
            rec.update(kind="weights+activations", bits=4, kernel=QuaRotLinear.kernel)
        elif method == "sparse24":
            for l, blk in enumerate(blocks(model)):
                for n in LINEARS:
                    setattr(blk, n, Sparse24Linear(art[f"{l}.{n}"].to(dt)))
            rec.update(kind="weights", bits=None, kernel=Sparse24Linear.kernel, sparsity="2:4")
        else:
            raise ValueError(method)
        del art
    torch.cuda.empty_cache()
    return rec


# -------------------------------------------------------------------------------------------------------- gate C1
@torch.no_grad()
def c1_check(model, method, n=100, layers=(0, 15, 31), tol=0.01, seed=0):
    """Gate C1: for each type of block linear (in 3 layers), 100 random inputs (standard normal, bf16); the relative
    error ||y - y_ref|| / ||y_ref|| of each input, against the reference implementation. Passes if every value is
    at most `tol`. For 2:4 sparsity, also the zeros in each group of 4 weights of every block linear."""
    dev = model.model.transformer.wte.weight.device
    g = torch.Generator(device=dev).manual_seed(seed)
    rows, worst = [], 0.0
    for l in layers:
        for name in LINEARS:
            m = getattr(blocks(model)[l], name)
            x = torch.randn(n, m.in_features, generator=g, device=dev, dtype=torch.bfloat16)
            if method == "awq4":
                y = torch.cat([m.k(x[i:i + 50]) for i in range(0, n, 50)]).float()       # the int4 kernel path
                ref = x.float() @ m.w_bf16.float().t()
            else:
                y, ref = m(x).float(), m.reference(x).float()
            rel = ((y - ref).norm(dim=1) / ref.norm(dim=1).clamp(min=1e-12))
            worst = max(worst, float(rel.max()))
            rows.append(dict(layer=l, linear=name, max_rel_err=float(rel.max()), mean_rel_err=float(rel.mean())))
    rep = dict(method=method, n_inputs=n, layers=list(layers), tol=tol, max_rel_err=worst, passed=worst <= tol,
               rows=rows)
    if method == "sparse24":
        cnt = {}
        for blk in blocks(model):
            for name in LINEARS:
                z = (getattr(blk, name).w_dense.reshape(-1, 4) == 0).sum(-1)
                for k in range(5):
                    cnt[k] = cnt.get(k, 0) + int((z == k).sum())
        rep["zeros_per_group_of_4"] = cnt
        rep["groups_with_at_least_2_zeros"] = sum(v for k, v in cnt.items() if k >= 2) / sum(cnt.values())
        rep["passed"] = rep["passed"] and cnt.get(0, 0) == 0 and cnt.get(1, 0) == 0
    return rep


def main():
    """The GPU job of gate C: make a calibrated compression (or none for fp8), save it, and run C1."""
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transformers import AutoTokenizer
    from model.modeling_llada import LLaDAModelLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--revision", default="08b83a6feb34df1a6011b80c3c00c7563e963b07")
    ap.add_argument("--art-dir", default="/project2/ppanda_1750/kook/dllm/models/phase3")
    ap.add_argument("--c1-out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    t0 = time.time()
    if a.method != "fp8" and os.path.exists(os.path.join(a.art_dir, f"{a.method}.safetensors")):
        print(f"[{a.method}] the artifact exists; it is not made again", flush=True)
    elif a.method != "fp8":
        mk = dict(awq4=make_awq, quarot_w4a4=make_quarot, sparse24=make_sparse24)[a.method]
        art, meta = mk(model, tok, dev)
        meta.update(model=a.model, model_revision=a.revision, minutes=round((time.time() - t0) / 60, 1))
        save(art, meta, a.art_dir, a.method)
        print(f"[{a.method}] saved to {a.art_dir} ({meta['minutes']} min)", flush=True)
        del art, model
        torch.cuda.empty_cache()
        model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True,
                                             torch_dtype=torch.bfloat16).to(dev).eval()
    rec = apply(model, a.method, a.art_dir)
    rep = c1_check(model, a.method)
    rep["compression"] = {k: v for k, v in rec.items() if k != "meta"}
    os.makedirs(os.path.dirname(a.c1_out), exist_ok=True)
    json.dump(rep, open(a.c1_out, "w"), indent=1)
    print(f"C1 {a.method}: max relative error {rep['max_rel_err']:.2e} (limit {rep['tol']}) -> "
          f"{'pass' if rep['passed'] else 'FAIL'}" + (f"; zeros per group of 4 {rep['zeros_per_group_of_4']}"
                                                     if a.method == "sparse24" else ""), flush=True)


if __name__ == "__main__":
    main()
