"""Phase 3 stage 3-0: the A compressions that Phase 3 adds (results/phase3/PREREG.md, section 3.1 and gate C of
section 2.3; docs/plan/07_phase3.md, section 3.2), for LLaDA-8B-Instruct and for the Qwen2-like models (Fast-dLLM v2
7B, Dream-v0-Instruct-7B). The other A compressions are older: the depth cut (KL-greedy set), GPTQ 4-bit of
LLaDA-8B-Instruct (the published checkpoint, `gptq_dequant.py`, Marlin) and SoloQ W4A4 (imported from outside the fork).

- fp8: FP8 E4M3 weights (one scale per output channel) and activations (one scale per token, computed at run time).
  Real kernel: torch._scaled_mm with row-wise scales, on every pass. No calibration.
- awq4: AWQ 4-bit weights, group 128, asymmetric (zero points), as llm-awq (github.com/mit-han-lab/llm-awq): the
  scale search (20 grid points) on the groups (norm1 -> q, k, v), (v -> o, only without grouped keys), (norm2 -> gate,
  up) and (up -> down), then the clip search (20 grid points, shrink up to 0.5; not on q and k). The calibration
  data: PileVal, 128 documents of at most 512 tokens, joined and cut into blocks of 512 (seed 42).
  Real kernel: torch's tinygemm int4 kernel (aten._weight_int4pack_mm) on calls with at most `row_max` rows (the
  refinement passes); a bf16 copy of the expanded weights on larger calls (the cache-writing passes), as Marlin.
- quarot_w4a4: QuaRot (github.com/spcl/QuaRot, the command of its README: --rotate --a_bits 4 --w_bits 4 --w_clip):
  the norms are folded into the next linears; a random Hadamard rotation of the residual stream; online Hadamard
  transforms before o and down (sizes a x 2^m, with a Paley Hadamard matrix of order a); weights 4-bit, per output
  channel, symmetric, GPTQ with the clip search (WikiText-2, 128 x 2048, damp 0.01, no act-order, true-sequential
  groups); activations 4-bit, per token, symmetric, clip ratio 1.0. The embedding and the LM head stay in bf16.
  Simulated (fake quantization), so its speed is a projection.
- sparse24: 2:4 sparsity with SparseGPT (github.com/IST-DASLab/sparsegpt, prunen 2, prunem 4, block 128, damp
  0.01; C4 train shard 0, 128 x 2048, seed 0). Real kernel: the semi-structured sparse tensor of PyTorch
  (torch.sparse.to_sparse_semi_structured) on every pass.
- gptq4 (the models with no published GPTQ checkpoint): GPTQ 4-bit, group 128, symmetric (zero point 8), damp 0.1,
  no act-order, true-sequential groups, C4 128 x 2048: the settings of the LLaDA-8B-Instruct checkpoint
  (GPTQModel 2.2.0). Real kernel: Marlin (`gptq_dequant.MarlinLinear`) on calls with at most `row_max` rows.

A calibrated compression is made once (`make`, in a GPU job) and saved to a folder; a cell loads it (`apply`).
Gate C1 (`c1_check`) compares each compressed linear with a reference implementation on 100 random inputs.
The class `Arch` maps the roles (q, k, v, o, gate, up, down; norm1, norm2) to the module names of each model. The
artifacts of LLaDA-8B-Instruct keep the LLaDA names (for example "0.q_proj" and "0.attn_out").
"""
import glob
import gzip
import inspect
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

ROLES = ("q", "k", "v", "o", "gate", "up", "down")
METHODS = ("fp8", "awq4", "quarot_w4a4", "sparse24", "gptq4")
FP8_MAX = 448.0
HUB = "/project2/ppanda_1750/kook/dllm/models/hub"


def _snap(repo):
    s = sorted(glob.glob(f"{HUB}/datasets--{repo.replace('/', '--')}/snapshots/*"))
    assert s, f"dataset {repo} is not in the cache"
    return s[0]


def _rget(obj, name):
    for p in name.split("."):
        obj = getattr(obj, p)
    return obj


def _rset(obj, name, val):
    *head, last = name.split(".")
    for p in head:
        obj = getattr(obj, p)
    setattr(obj, last, val)


class Arch:
    """The module names of a model family. kind "llada": LLaDA-8B-Instruct (model.model.transformer); kind "qwen2":
    the Qwen2-like models (model.model.layers: self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj)."""

    def __init__(self, model):
        inner = model.model
        if hasattr(inner, "transformer"):
            tr = inner.transformer
            self.kind, self.blocks, self.embed, self.final_norm = "llada", tr.blocks, tr.wte, tr.ln_f
            self._head = (tr, "ff_out")
            self.names = dict(q="q_proj", k="k_proj", v="v_proj", o="attn_out", gate="ff_proj", up="up_proj",
                              down="ff_out")
            self.norm1, self.norm2 = "attn_norm", "ff_norm"
        else:
            self.kind, self.blocks, self.embed, self.final_norm = "qwen2", inner.layers, inner.embed_tokens, inner.norm
            self._head = (model, "lm_head")
            self.names = dict(q="self_attn.q_proj", k="self_attn.k_proj", v="self_attn.v_proj", o="self_attn.o_proj",
                              gate="mlp.gate_proj", up="mlp.up_proj", down="mlp.down_proj")
            self.norm1, self.norm2 = "input_layernorm", "post_attention_layernorm"

    @property
    def head(self):
        return getattr(*self._head)

    def lin(self, blk, role):
        return _rget(blk, self.names[role])

    def set_lin(self, blk, role, mod):
        _rset(blk, self.names[role], mod)

    def norm(self, blk, which):
        return getattr(blk, self.norm1 if which == 1 else self.norm2)

    def key(self, l, role):
        return f"{l}.{self.names[role]}"

    def attn_out(self, blk, xn, kw):
        """The output of the attention sub-block (with o) for normalized inputs xn."""
        if self.kind == "llada":
            return blk.attention(blk.q_proj(xn), blk.k_proj(xn), blk.v_proj(xn), None)[0]
        sig = inspect.signature(blk.self_attn.forward).parameters
        o = blk.self_attn(hidden_states=xn, **{k: v for k, v in kw.items() if k in sig})
        return o[0] if isinstance(o, tuple) else o

    def mlp_out(self, blk, xn):
        if self.kind == "llada":
            return blk.ff_out(blk.act(blk.ff_proj(xn)) * blk.up_proj(xn))
        return blk.mlp(xn)


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
    """SparseGPT and GPTQ (datautils.get_c4): random documents of the train shard 0 with more than seqlen tokens."""
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
class _Stop(Exception):
    pass


class LayerRunner:
    """The inputs of block 0 for the calibration ids, and a call of any block on them. LLaDA blocks take the hidden
    states only. For the other models, the keyword arguments of the model's own call of block 0 (attention mask,
    rotary terms) are caught once and reused, so every block runs as in the model's forward (with use_cache=False)."""

    def __init__(self, model, arch, ids, dev):
        self.arch = arch
        n = ids.shape[0]
        self.bs = next(b for b in (4, 2, 1) if n % b == 0)
        with torch.no_grad():
            if arch.kind == "llada":
                self.kw = {}
                self.X = arch.embed(ids.to(dev))
                return
            caught = {}

            def hook(mod, args, kwargs):
                caught["x"] = args[0] if args else kwargs["hidden_states"]
                caught["kw"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
                raise _Stop

            h = arch.blocks[0].register_forward_pre_hook(hook, with_kwargs=True)
            xs = []
            for i in range(0, n, self.bs):
                try:
                    model(ids[i:i + self.bs].to(dev), use_cache=False)
                except _Stop:
                    pass
                xs.append(caught["x"])
            h.remove()
            self.X, self.kw = torch.cat(xs), caught["kw"]

    @torch.no_grad()
    def run(self, blk, X):
        out = torch.empty_like(X)
        for i in range(0, X.shape[0], self.bs):
            o = blk(X[i:i + self.bs], **self.kw)
            out[i:i + self.bs] = o[0] if isinstance(o, tuple) else o
        return out


class InputCapture:
    """Forward pre-hooks that collect the input of named linears: a Hessian sum (GPTQ, SparseGPT) or the raw rows."""

    def __init__(self, mods, mode="hessian"):
        self.mode, self.store, self.hooks = mode, {}, []
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


def act_quant_sym(x, bits=4, clip=1.0, out_dtype=None):
    """QuaRot's ActQuantizer: symmetric, one scale per token (row), levels -(2^(b-1)) .. 2^(b-1)-1.
    Computed in float64: a bf16 input often falls exactly on a rounding tie (x / s = k + 1/2), and float32 breaks
    such ties differently from the fp64 reference of gate C1 (a change of one level, about 1 % of the output)."""
    maxq = 2 ** (bits - 1) - 1
    xf = x.double()
    s = (xf.abs().amax(-1, keepdim=True) * clip / maxq).clamp(min=1e-8)
    return (torch.clamp(torch.round(xf / s), -(maxq + 1), maxq) * s).to(out_dtype or x.dtype)


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


def gptq_sym_group(W, H, bits=4, group=128, percdamp=0.1, blocksize=128):
    """GPTQ (fasterquant) with groups of `group` input columns, symmetric as GPTQModel (levels 0..15, zero point 8,
    scale = 2 max|w| / 15 of the group, set when the group starts, from the updated weights). Returns the dequantized
    weight and the scales (groups x out), the layout of a GPTQ checkpoint."""
    W = W.float().clone()
    maxq = 2 ** bits - 1
    zero = (maxq + 1) / 2
    Hinv, dead = _hinv(H, percdamp)
    W[:, dead] = 0
    out_f, ncol = W.shape
    Q = torch.zeros_like(W)
    scales = torch.zeros(ncol // group, out_f, device=W.device)
    scale = None
    for i1 in range(0, ncol, blocksize):
        i2 = min(i1 + blocksize, ncol)
        W1 = W[:, i1:i2].clone()
        Q1, E1 = torch.zeros_like(W1), torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if c % group == 0:
                g = torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :group]
                xmax = torch.maximum(g.abs().amax(1), torch.full((out_f,), 1e-8, device=W.device))
                scale = (2 * xmax / maxq).half().float()            # the fp16 scale that Marlin stores
                scales[c // group] = scale
            w = W1[:, i]
            q = (torch.clamp(torch.round(w / scale) + zero, 0, maxq) - zero) * scale
            Q1[:, i] = q
            e = (w - q) / Hi[i, i]
            W1[:, i:] -= e[:, None] @ Hi[i, i:][None, :]
            E1[:, i] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, scales


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


def _is_prime(q):
    return q > 1 and all(q % d for d in range(2, int(q ** 0.5) + 1))


def _jacobsthal(q):
    res = {(i * i) % q for i in range(1, q)}
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in res else -1)                        # noqa: E731
    return torch.tensor([[chi(j - i) for j in range(q)] for i in range(q)], dtype=torch.float64)


def paley_hadamard(a):
    """A Hadamard matrix of order a: Paley I (a - 1 prime, = 3 mod 4) or Paley II (a / 2 - 1 prime, = 1 mod 4)."""
    if _is_prime(a - 1) and (a - 1) % 4 == 3:
        q = a - 1
        S = torch.zeros(q + 1, q + 1, dtype=torch.float64)
        S[0, 1:], S[1:, 0], S[1:, 1:] = 1, -1, _jacobsthal(q)
        H = S + torch.eye(q + 1, dtype=torch.float64)
    elif a % 2 == 0 and _is_prime(a // 2 - 1) and (a // 2 - 1) % 4 == 1:
        q = a // 2 - 1
        C = torch.zeros(q + 1, q + 1, dtype=torch.float64)
        C[0, 1:], C[1:, 0], C[1:, 1:] = 1, 1, _jacobsthal(q)
        A = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
        B = torch.tensor([[1.0, -1.0], [-1.0, -1.0]], dtype=torch.float64)
        H = torch.kron(C, A) + torch.kron(torch.eye(q + 1, dtype=torch.float64), B)
    else:
        raise ValueError(f"no Paley construction of order {a}")
    assert torch.allclose(H @ H.t(), a * torch.eye(a, dtype=torch.float64)), a
    return H


def hadamard_factors(n):
    """n = a * m with m a power of two and a = 1 or a Paley order: the two normalized factors (H_a, H_m)."""
    for a in (1, 12, 20, 28, 36, 44, 52, 60, 76, 84, 108, 148, 172):
        if n % a == 0 and (n // a) & (n // a - 1) == 0:
            m = n // a
            HA = torch.ones(1, 1, dtype=torch.float64) if a == 1 else paley_hadamard(a)
            return HA / math.sqrt(a), _sylvester(m) / math.sqrt(m)
    raise ValueError(f"no Hadamard construction for size {n}")


class Hadamard:
    """x -> x @ (H_a kron H_m), applied as H_a^T X H_m on x reshaped to (a, m)."""

    def __init__(self, n, dev):
        Ha, Hm = hadamard_factors(n)
        self.n, self.a, self.m = n, Ha.shape[0], Hm.shape[0]
        self.Ha, self.Hm = Ha.float().to(dev), Hm.float().to(dev)

    def __call__(self, x, out_dtype=None):
        shp = x.shape
        X = x.float().reshape(-1, self.a, self.m) @ self.Hm
        if self.a > 1:
            X = torch.einsum("ik,bim->bkm", self.Ha, X)
        return X.reshape(shp).to(out_dtype or x.dtype)

    def matrix(self):
        return torch.kron(self.Ha.double(), self.Hm.double())


# ----------------------------------------------------------------------------------------------- compressed linears
def _bias(lin):
    return None if lin.bias is None else lin.bias.data.clone()


class FP8Linear(torch.nn.Module):
    """FP8 E4M3 weights (one scale per output channel) and activations (one scale per token, at run time);
    torch._scaled_mm with row-wise scales, bf16 output. Rows are padded to a multiple of 16."""

    kernel = "torch._scaled_mm (FP8 E4M3, row-wise scales)"

    def __init__(self, W, bias=None):
        super().__init__()
        self.out_features, self.in_features = W.shape
        s = (W.float().abs().amax(1, keepdim=True) / FP8_MAX).clamp(min=1e-12)          # (out, 1)
        self.register_buffer("wq", (W.float() / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn))
        self.register_buffer("ws", s.t().contiguous())                                   # (1, out)
        self.register_buffer("bias", bias)

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
        y = torch._scaled_mm(xq, self.wq.t(), scale_a=xs, scale_b=self.ws, out_dtype=torch.bfloat16)[:M]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], self.out_features)

    def reference(self, x):
        """The fake-quantization reference (fp32): the same FP8 values, dequantized, and a fp32 matmul."""
        xq, xs = self.quant_x(x.reshape(-1, self.in_features))
        y = (xq.float() * xs) @ (self.wq.float() * self.ws.t()).t()
        return y if self.bias is None else y + self.bias.float()


class Sparse24Linear(torch.nn.Module):
    """2:4 sparse weights in PyTorch's semi-structured sparse format; F.linear runs the sparse kernel."""

    kernel = "torch semi-structured sparse (2:4)"

    def __init__(self, W, bias=None):
        super().__init__()
        from torch.sparse import to_sparse_semi_structured
        self.out_features, self.in_features = W.shape
        self.register_buffer("w_dense", W)                                               # for C1 and the record
        self.register_buffer("bias", bias)
        self.w = to_sparse_semi_structured(W)

    def forward(self, x):
        shp = x.shape
        y = F.linear(x.reshape(-1, self.in_features), self.w)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], self.out_features)

    def reference(self, x):
        y = x.reshape(-1, self.in_features).float() @ self.w_dense.float().t()
        return y if self.bias is None else y + self.bias.float()


class QuaRotLinear(torch.nn.Module):
    """QuaRot fake quantization: (online Hadamard), 4-bit per-token symmetric activations, 4-bit weights (stored
    dequantized), bf16 matmul."""

    kernel = "none (simulated)"

    def __init__(self, Wq, had=None, bias=None, a_bits=4, a_clip=1.0):
        super().__init__()
        self.out_features, self.in_features = Wq.shape
        self.register_buffer("w", Wq)
        self.register_buffer("bias", bias)
        self.had, self.a_bits, self.a_clip = had, a_bits, a_clip
        self.quant_act = True           # off while GPTQ collects its Hessians (QuaRot quantizes activations later)
        self.capture = None             # GPTQ: a function that receives the input after the online Hadamard

    def forward(self, x):
        dt = x.dtype
        if self.had is not None:
            x = self.had(x, torch.float32)  # the quantizer reads the fp32 output (QuaRot's kernel: fp16, not bf16)
        if self.capture is not None:
            self.capture(x.to(dt))          # GPTQ sees the bf16 values, as when the artifacts were made (ee2f5b2)
        x = act_quant_sym(x, self.a_bits, self.a_clip, dt) if self.quant_act else x.to(dt)
        return F.linear(x, self.w, self.bias)

    def reference(self, x):
        """fp64: the Hadamard as one dense matrix, then the same quantizer formula, then the matmul."""
        x2 = x.reshape(-1, self.in_features).double()
        if self.had is not None:
            x2 = x2 @ self.had.matrix().to(x2.device)
        maxq = 2 ** (self.a_bits - 1) - 1
        s = (x2.abs().amax(-1, keepdim=True) * self.a_clip / maxq).clamp(min=1e-8)
        xq = torch.clamp(torch.round(x2 / s), -(maxq + 1), maxq) * s
        y = xq.to(torch.bfloat16).double() @ self.w.double().t()
        return y if self.bias is None else y + self.bias.double()


class KernelLinear(torch.nn.Module):
    """A 4-bit kernel module (tinygemm int4 or Marlin) on calls with at most `row_max` rows, a bf16 copy of the
    expanded weights on larger calls, and the bias (if any) after either path."""

    def __init__(self, deq, kernel_mod, kernel_name, row_max, bias=None):
        super().__init__()
        self.out_features, self.in_features = deq.shape
        self.register_buffer("w_bf16", deq)
        self.register_buffer("bias", bias)
        self.k, self.kernel, self.row_max = kernel_mod, kernel_name, row_max

    def forward(self, x):
        if x.numel() // self.in_features > self.row_max:
            y = F.linear(x, self.w_bf16.to(x.dtype))
        else:
            y = self.k(x)
        return y if self.bias is None else y + self.bias

    def kernel_call(self, x):
        y = self.k(x)
        return y if self.bias is None else y + self.bias

    def reference(self, x):
        y = x.reshape(-1, self.in_features).float() @ self.w_bf16.float().t()
        return y if self.bias is None else y + self.bias.float()


def awq_linear(codes, scales, zeros, dtype, row_max=64, bias=None):
    """AWQ weights on torch's tinygemm int4 kernel (`gptq_dequant.Int4PackLinear`)."""
    import gptq_dequant as gq
    out_f, in_f = codes.shape
    deq = ((codes.float().reshape(out_f, in_f // 128, 128) - zeros.float()[..., None]) * scales.float()[..., None])
    deq = deq.reshape(out_f, in_f).to(dtype)
    k = gq.Int4PackLinear(codes.int(), scales.float().t().contiguous(), zeros.float().t().contiguous(), out_f, in_f,
                          dtype)
    return KernelLinear(deq, k, f"aten._weight_int4pack_mm (tinygemm int4, group 128) on <= {row_max} rows; bf16 "
                                f"above", row_max, bias)


def marlin_linear(deq, scales, row_max=64, bias=None):
    """Symmetric group-128 4-bit weights (zero point 8) on Marlin (`gptq_dequant.MarlinLinear` packs them)."""
    import gptq_dequant as gq
    zeros = torch.full_like(scales, 8.0)
    return _MarlinWithBias(gq.MarlinLinear(deq, None, scales.half(), zeros, row_max), bias, row_max)


class _MarlinWithBias(KernelLinear):
    def __init__(self, m, bias, row_max):
        torch.nn.Module.__init__(self)
        self.out_features, self.in_features = m.out_features, m.in_features
        self.m = m
        self.register_buffer("bias", bias)
        self.row_max = row_max
        self.kernel = f"Marlin (fp16 x int4, group 128) on <= {row_max} rows; bf16 above"

    @property
    def w_bf16(self):
        return self.m.w_bf16

    def forward(self, x):
        y = self.m(x)
        return y if self.bias is None else y + self.bias

    def kernel_call(self, x):
        y = self.m.m(x.half()).to(x.dtype)
        return y if self.bias is None else y + self.bias


# ------------------------------------------------------------------------------------------------------- AWQ
@torch.no_grad()
def _awq_search(x, linears, inspect_fn, n_grid=20):
    """llm-awq auto_scale._search_module_scale: s = mean|x|^alpha, normalized by sqrt(max * min), alpha = r / 20;
    the weights are fake-quantized (4-bit, group 128, asymmetric); the smallest MSE of the inspected output wins.
    The weights are restored after each trial."""
    x_max = x.abs().reshape(-1, x.shape[-1]).float().mean(0)
    org_w = [l.weight.data.clone() for l in linears]
    org_out = inspect_fn(x)
    best, best_s, hist = float("inf"), None, []
    for r in range(n_grid):
        s = x_max.pow(r / n_grid).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        for l, w in zip(linears, org_w):
            l.weight.data = (asym_quant_groups(w.float() * s[None, :])[0] / s[None, :]).to(w.dtype)
        loss = (org_out.float() - inspect_fn(x).float()).pow(2).mean().item()
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


def _scale_cols(lin, s):
    lin.weight.data = (lin.weight.float() * s[None, :]).to(lin.weight.dtype)


def _scale_rows(lin, s):
    lin.weight.data = (lin.weight.float() / s[:, None]).to(lin.weight.dtype)
    if lin.bias is not None:
        lin.bias.data = (lin.bias.float() / s).to(lin.bias.dtype)


@torch.no_grad()
def make_awq(model, tok, dev, log=print):
    """llm-awq run_awq: for each block, the input features of the linears and the next block's inputs come from the
    unchanged block; the scale searches run on the unchanged weights; then the scales are applied (and the stored
    features divided by them); then the clip search (not on q and k)."""
    A = Arch(model)
    ids = calib_pileval(tok)
    log(f"[awq] calibration: PileVal, {tuple(ids.shape)} tokens")
    runner = LayerRunner(model, A, ids, dev)
    X = runner.X
    for l, blk in enumerate(A.blocks):
        t0 = time.time()
        roles = ("q", "o", "gate", "down")
        cap = InputCapture({r: A.lin(blk, r) for r in roles}, mode="rows")
        Y = runner.run(blk, X)
        cap.remove()
        f = {r: cap.rows(r).reshape(X.shape[0], X.shape[1], -1) for r in roles}
        del cap

        def attn(xn):
            o = torch.empty(*xn.shape[:2], A.lin(blk, "o").out_features, device=dev, dtype=xn.dtype)
            for i in range(0, xn.shape[0], runner.bs):
                o[i:i + runner.bs] = A.attn_out(blk, xn[i:i + runner.bs], runner.kw)
            return o

        q, k, v, o_, gate, up, down = (A.lin(blk, r) for r in ROLES)
        mha = v.out_features == o_.in_features and v.out_features == o_.out_features
        s1, l1, a1 = _awq_search(f["q"], [q, k, v], attn)
        s2, l2, a2 = _awq_search(f["o"], [o_], o_) if mha else (None, float("nan"), float("nan"))
        s3, l3, a3 = _awq_search(f["gate"], [gate, up], lambda xn: A.mlp_out(blk, xn))
        s4, l4, a4 = _awq_search(f["down"], [down], down)
        n1, n2 = A.norm(blk, 1), A.norm(blk, 2)
        n1.weight.data = (n1.weight.float() / s1).to(n1.weight.dtype)
        for lin in (q, k, v):
            _scale_cols(lin, s1)
        if mha:
            _scale_rows(v, s2)
            _scale_cols(o_, s2)
        n2.weight.data = (n2.weight.float() / s3).to(n2.weight.dtype)
        for lin in (gate, up):
            _scale_cols(lin, s3)
        _scale_rows(up, s4)
        _scale_cols(down, s4)
        xin = {"v": f["q"].float() / s1, "o": f["o"].float() / (s2 if mha else 1.0),
               "gate": f["gate"].float() / s3, "up": f["gate"].float() / s3, "down": f["down"].float() / s4}
        del f
        for r, xi in xin.items():
            _awq_clip(A.lin(blk, r), xi)
        del xin
        X = Y
        torch.cuda.empty_cache()
        log(f"[awq] block {l}: alpha qkv {a1:.2f} v-o {a2:.2f} mlp {a3:.2f} up-down {a4:.2f}; "
            f"mse {l1:.2e} {l2:.2e} {l3:.2e} {l4:.2e} ({time.time() - t0:.0f} s)")
    art = {}
    for l, blk in enumerate(A.blocks):
        for r in ROLES:
            lin = A.lin(blk, r)
            _, q, s, z = asym_quant_groups(lin.weight.data)
            key = A.key(l, r)
            art[f"{key}.codes"], art[f"{key}.scales"], art[f"{key}.zeros"] = q.cpu(), s.float().cpu(), z.cpu()
            if lin.bias is not None:
                art[f"{key}.bias"] = lin.bias.data.cpu()
        art[f"{l}.{A.norm1}"] = A.norm(blk, 1).weight.data.cpu()
        art[f"{l}.{A.norm2}"] = A.norm(blk, 2).weight.data.cpu()
    return art, dict(calibration="mit-han-lab/pile-val-backup val.jsonl.zst, shuffle seed 42, 128 documents of at "
                                 "most 512 tokens, joined, blocks of 512", calib_shape=list(ids.shape),
                     bits=4, group=128, zero_point=True, scale_grid=20, clip_grid=20, clip_max_shrink=0.5)


# --------------------------------------------------------------------------------------------------------- QuaRot
@torch.no_grad()
def quarot_rotate(model, seed=0):
    """Fold the RMS norm weights into the next linears; rotate the residual stream by Q = diag(random signs) H
    (QuaRot random_hadamard_matrix, seed); fold the online Hadamards of o (its input width) and down (the MLP width)
    into their weights (W <- W H). With the online transforms of `quarot_wrap`, the function does not change."""
    A = Arch(model)
    dev, dt = A.embed.weight.device, A.embed.weight.dtype
    d = A.embed.weight.shape[1]

    def fold(norm, lins):
        g = norm.weight.data.double()
        for lin in lins:
            lin.weight.data = (lin.weight.data.double() * g[None, :]).to(dt)
        norm.weight.data.fill_(1.0)

    for blk in A.blocks:
        fold(A.norm(blk, 1), [A.lin(blk, r) for r in ("q", "k", "v")])
        fold(A.norm(blk, 2), [A.lin(blk, r) for r in ("gate", "up")])
    fold(A.final_norm, [A.head])
    gen = torch.Generator().manual_seed(seed)
    sgn = torch.randint(0, 2, (d,), generator=gen).double() * 2 - 1
    Q = (sgn[:, None] * Hadamard(d, "cpu").matrix()).to(dev)                                # diag(s) @ H
    A.embed.weight.data = (A.embed.weight.data.double() @ Q).to(dt)
    for blk in A.blocks:
        for r in ("q", "k", "v", "gate", "up"):
            lin = A.lin(blk, r)
            lin.weight.data = (lin.weight.data.double() @ Q).to(dt)
        for r in ("o", "down"):
            lin = A.lin(blk, r)
            assert lin.bias is None, f"the output side of {r} has a bias"
            lin.weight.data = (Q.t() @ lin.weight.data.double()).to(dt)
    A.head.weight.data = (A.head.weight.data.double() @ Q).to(dt)
    blk0 = A.blocks[0]
    had_o, had_d = Hadamard(A.lin(blk0, "o").in_features, dev), Hadamard(A.lin(blk0, "down").in_features, dev)
    for blk in A.blocks:
        A.lin(blk, "o").weight.data = had_o(A.lin(blk, "o").weight.data.double()).to(dt)
        A.lin(blk, "down").weight.data = had_d(A.lin(blk, "down").weight.data.double()).to(dt)
    return had_o, had_d


def quarot_wrap(model, had_o, had_d):
    """Replace each block linear by a QuaRotLinear (online Hadamard before o and down)."""
    A = Arch(model)
    for blk in A.blocks:
        for r in ROLES:
            lin = A.lin(blk, r)
            had = had_o if r == "o" else (had_d if r == "down" else None)
            A.set_lin(blk, r, QuaRotLinear(lin.weight.data, had, _bias(lin)))


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
    out = model(ids, use_cache=False) if Arch(model).kind == "qwen2" else model(ids)
    return out.logits.float()


GROUPS = [("q", "k", "v"), ("o",), ("gate", "up"), ("down",)]


@torch.no_grad()
def make_quarot(model, tok, dev, log=print, seed=0):
    A = Arch(model)
    ids = calib_wikitext2(tok)
    log(f"[quarot] calibration: WikiText-2 train, {tuple(ids.shape)} tokens")
    probe = ids[:2, :256].to(dev)
    ref = _logits(model, probe)
    had_o, had_d = quarot_rotate(model, seed)
    quarot_wrap(model, had_o, had_d)
    for blk in A.blocks:
        for r in ROLES:
            A.lin(blk, r).quant_act = False
    # the rotated model with its online Hadamards and no quantization must compute the same function
    rot = _logits(model, probe)
    inv = float((rot - ref).norm() / ref.norm())
    log(f"[quarot] rotation check: relative logit difference {inv:.3e} (no quantization, 2 x 256 tokens)")
    runner = LayerRunner(model, A, ids, dev)
    X = runner.X
    for l, blk in enumerate(A.blocks):
        t0 = time.time()
        for g in GROUPS:
            cap = _QCapture({r: A.lin(blk, r) for r in g})
            runner.run(blk, X)
            cap.remove()
            for r in g:
                m = A.lin(blk, r)
                m.w.data = gptq_sym_perchannel(m.w.data, cap.hessian(r)).to(m.w.dtype)
            del cap
        X = runner.run(blk, X)
        torch.cuda.empty_cache()
        log(f"[quarot] block {l}: GPTQ done ({time.time() - t0:.0f} s)")
    for blk in A.blocks:
        for r in ROLES:
            A.lin(blk, r).quant_act = True
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return sd, dict(calibration="Salesforce/wikitext wikitext-2-raw-v1 train, joined with blank lines, 128 random "
                                "windows of 2048 tokens, seed 0", rotation_seed=seed, rotation_check_rel_logit=inv,
                    w_bits=4, w_scheme="per output channel, symmetric, GPTQ (damp 0.01, block 128, no act-order, "
                                       "true-sequential groups), MSE clip search (norm 2.4, grid 100, shrink 0.8)",
                    a_bits=4, a_scheme="per token, symmetric, clip 1.0", online_hadamard=dict(
                        o=[had_o.a, had_o.m], down=[had_d.a, had_d.m]))


# ------------------------------------------------------------------------------------------------------ SparseGPT
@torch.no_grad()
def make_sparse24(model, tok, dev, log=print):
    A = Arch(model)
    ids = calib_c4(tok)
    log(f"[sparsegpt] calibration: C4 train shard 0, {tuple(ids.shape)} tokens")
    runner = LayerRunner(model, A, ids, dev)
    X = runner.X
    for l, blk in enumerate(A.blocks):
        t0 = time.time()
        cap = InputCapture({r: A.lin(blk, r) for r in ROLES})
        runner.run(blk, X)
        cap.remove()
        for r in ROLES:
            lin = A.lin(blk, r)
            lin.weight.data = sparsegpt_24(lin.weight.data, cap.hessian(r)).to(lin.weight.dtype)
        del cap
        X = runner.run(blk, X)
        torch.cuda.empty_cache()
        log(f"[sparsegpt] block {l}: pruned ({time.time() - t0:.0f} s)")
    art = {}
    for l, blk in enumerate(A.blocks):
        for r in ROLES:
            lin = A.lin(blk, r)
            art[A.key(l, r)] = lin.weight.data.cpu()
            if lin.bias is not None:
                art[f"{A.key(l, r)}.bias"] = lin.bias.data.cpu()
    return art, dict(calibration="allenai/c4 en/c4-train.00000-of-01024.json.gz, 128 random windows of 2048 tokens "
                                 "from documents longer than 2048 tokens, seed 0", calib_shape=list(ids.shape),
                     prunen=2, prunem=4, blocksize=128, percdamp=0.01)


# ----------------------------------------------------------------------------------------------------------- GPTQ
@torch.no_grad()
def make_gptq(model, tok, dev, log=print):
    """GPTQ 4-bit with the settings of the LLaDA-8B-Instruct checkpoint (GPTQModel 2.2.0: C4, group 128, symmetric,
    damp 0.1, no act-order, true-sequential)."""
    A = Arch(model)
    ids = calib_c4(tok)
    log(f"[gptq] calibration: C4 train shard 0, {tuple(ids.shape)} tokens")
    runner = LayerRunner(model, A, ids, dev)
    X = runner.X
    art = {}
    for l, blk in enumerate(A.blocks):
        t0 = time.time()
        for g in GROUPS:
            cap = InputCapture({r: A.lin(blk, r) for r in g})
            runner.run(blk, X)
            cap.remove()
            for r in g:
                lin = A.lin(blk, r)
                Wq, sc = gptq_sym_group(lin.weight.data, cap.hessian(r))
                lin.weight.data = Wq.to(lin.weight.dtype)
                art[f"{A.key(l, r)}.weight"] = lin.weight.data.cpu()
                art[f"{A.key(l, r)}.scales"] = sc.half().cpu()
                if lin.bias is not None:
                    art[f"{A.key(l, r)}.bias"] = lin.bias.data.cpu()
            del cap
        X = runner.run(blk, X)
        torch.cuda.empty_cache()
        log(f"[gptq] block {l}: done ({time.time() - t0:.0f} s)")
    return art, dict(calibration="allenai/c4 en/c4-train.00000-of-01024.json.gz, 128 random windows of 2048 tokens, "
                                 "seed 0", calib_shape=list(ids.shape), bits=4, group=128, sym=True, zero_point=8,
                     percdamp=0.1, act_order=False, true_sequential=True)


# ------------------------------------------------------------------------------------------------- save and apply
def save(art, meta, out_dir, method):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    save_file({k: v.contiguous() for k, v in art.items()}, os.path.join(out_dir, f"{method}.safetensors"))
    json.dump(meta, open(os.path.join(out_dir, f"{method}.json"), "w"), indent=1)


@torch.no_grad()
def apply(model, method, art_dir=None, row_max=64):
    """Put a compression into a loaded bf16 model. Returns the record of the compression for summary.json.
    `row_max` (awq4, gptq4): calls with more rows use the bf16 copy (the runners pass --marlin-row-max)."""
    from safetensors.torch import load_file
    A = Arch(model)
    dev, dt = A.embed.weight.device, A.embed.weight.dtype
    rec = dict(kind=None, method=method, artifact=None, kernel=None)
    if method == "fp8":
        for blk in A.blocks:
            for r in ROLES:
                lin = A.lin(blk, r)
                A.set_lin(blk, r, FP8Linear(lin.weight.data, _bias(lin)))
        rec.update(kind="weights+activations", bits=8, kernel=FP8Linear.kernel,
                   scheme="E4M3; weights: one scale per output channel; activations: one scale per token at run time")
        torch.cuda.empty_cache()
        return rec
    path = os.path.join(art_dir, f"{method}.safetensors")
    art = load_file(path, device=str(dev))
    meta = json.load(open(os.path.join(art_dir, f"{method}.json")))
    rec.update(artifact=path, meta=meta)
    if method == "awq4":
        for l, blk in enumerate(A.blocks):
            A.norm(blk, 1).weight.data.copy_(art[f"{l}.{A.norm1}"])
            A.norm(blk, 2).weight.data.copy_(art[f"{l}.{A.norm2}"])
            for r in ROLES:
                key = A.key(l, r)
                A.set_lin(blk, r, awq_linear(art[f"{key}.codes"], art[f"{key}.scales"], art[f"{key}.zeros"], dt,
                                             row_max, art.get(f"{key}.bias")))
        rec.update(kind="weights", bits=4, kernel=A.lin(A.blocks[0], "q").kernel, row_max=row_max)
    elif method == "gptq4":
        for l, blk in enumerate(A.blocks):
            for r in ROLES:
                key = A.key(l, r)
                A.set_lin(blk, r, marlin_linear(art[f"{key}.weight"].to(dt), art[f"{key}.scales"], row_max,
                                                art.get(f"{key}.bias")))
        rec.update(kind="weights", bits=4, kernel=A.lin(A.blocks[0], "q").kernel, row_max=row_max)
    elif method == "quarot_w4a4":
        hd = meta["online_hadamard"]
        ho, hdn = hd.get("o", hd.get("attn_out")), hd.get("down", hd.get("ff_out"))
        quarot_wrap(model, Hadamard(ho[0] * ho[1], dev), Hadamard(hdn[0] * hdn[1], dev))
        model.load_state_dict(art, strict=True)
        rec.update(kind="weights+activations", bits=4, kernel=QuaRotLinear.kernel)
    elif method == "sparse24":
        for l, blk in enumerate(A.blocks):
            for r in ROLES:
                key = A.key(l, r)
                A.set_lin(blk, r, Sparse24Linear(art[key].to(dt), art.get(f"{key}.bias")))
        rec.update(kind="weights", bits=None, kernel=Sparse24Linear.kernel, sparsity="2:4")
    else:
        raise ValueError(method)
    del art
    torch.cuda.empty_cache()
    return rec


# -------------------------------------------------------------------------------------------------------- gate C1
@torch.no_grad()
def c1_check(model, method, n=100, layers=None, tol=0.01, seed=0):
    """Gate C1: for each type of block linear (in 3 layers), 100 random inputs (standard normal, bf16); the relative
    error ||y - y_ref|| / ||y_ref|| of each input, against the reference implementation. Passes if every value is
    at most `tol`. The 4-bit kernels are called directly (their kernel path). For 2:4 sparsity, also the zeros in
    each group of 4 weights of every block linear."""
    A = Arch(model)
    dev = A.embed.weight.device
    L = len(A.blocks)
    layers = layers or (0, L // 2 - 1, L - 1)
    g = torch.Generator(device=dev).manual_seed(seed)
    rows, worst = [], 0.0
    for l in layers:
        for r in ROLES:
            m = A.lin(A.blocks[l], r)
            x = torch.randn(n, m.in_features, generator=g, device=dev, dtype=torch.bfloat16)
            if isinstance(m, KernelLinear):
                y = torch.cat([m.kernel_call(x[i:i + 50]) for i in range(0, n, 50)]).float()
            else:
                y = m(x).float()
            ref = m.reference(x).float()
            rel = ((y - ref).norm(dim=1) / ref.norm(dim=1).clamp(min=1e-12))
            worst = max(worst, float(rel.max()))
            rows.append(dict(layer=l, linear=A.names[r], max_rel_err=float(rel.max()), mean_rel_err=float(rel.mean())))
    rep = dict(method=method, n_inputs=n, layers=list(layers), tol=tol, max_rel_err=worst, passed=worst <= tol,
               rows=rows)
    if method == "sparse24":
        cnt = {}
        for blk in A.blocks:
            for r in ROLES:
                z = (A.lin(blk, r).w_dense.reshape(-1, 4) == 0).sum(-1)
                for k in range(5):
                    cnt[k] = cnt.get(k, 0) + int((z == k).sum())
        rep["zeros_per_group_of_4"] = cnt
        rep["passed"] = rep["passed"] and cnt.get(0, 0) == 0 and cnt.get(1, 0) == 0
    return rep


def load_model(name, dev):
    """LLaDA-8B-Instruct with the fork's model code; the other models with their own remote code."""
    import sys
    if "LLaDA-8B" in name:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from model.modeling_llada import LLaDAModelLM
        return LLaDAModelLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
        dev).eval()


def main():
    """The GPU job of gate C: make a calibrated compression (or none for fp8), save it, load it again, and run C1."""
    import argparse
    from transformers import AutoTokenizer
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
    model = load_model(a.model, dev)
    t0 = time.time()
    if a.method != "fp8" and os.path.exists(os.path.join(a.art_dir, f"{a.method}.safetensors")):
        print(f"[{a.method}] the artifact exists; it is not made again", flush=True)
    elif a.method != "fp8":
        mk = dict(awq4=make_awq, quarot_w4a4=make_quarot, sparse24=make_sparse24, gptq4=make_gptq)[a.method]
        art, meta = mk(model, tok, dev)
        meta.update(model=a.model, model_revision=a.revision, minutes=round((time.time() - t0) / 60, 1))
        save(art, meta, a.art_dir, a.method)
        print(f"[{a.method}] saved to {a.art_dir} ({meta['minutes']} min)", flush=True)
        del art, model
        torch.cuda.empty_cache()
        model = load_model(a.model, dev)
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
