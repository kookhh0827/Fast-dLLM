"""Phase 0 step 2 / runbook section 0a step 1 -- per-pass latency split for LLaDA-8B.

Produces the {layers / lm_head / softmax-confidence / sampling-indexing / python-other}
split for ONE refinement pass and ONE cache-writing pass, separately for each cache mode,
and the derived layer share f. `docs/06_phase_runbook.md` section 0a step 2 then asks a
single question of the output: does the excess over the layer-bandwidth floor sit INSIDE
the layer stack (f high, the method survives) or OUTSIDE it (f at its lower bound, no
depth schedule reaches the bar)?

Method. The reference generation functions in `generate.py` are called UNMODIFIED; the
instrumentation is hooks plus two wrappers, so the thing measured is the thing that ships:

  * every `model.forward` call is one pass. Wall time between the end of the previous pass
    and the end of this one is T_step: it contains the pass itself and all the sampling
    work that follows it.
  * CUDA events around each transformer block give T_layers.
  * T_model_gpu - T_layers is the embedding + final norm + LM head (the LM head dominates:
    126464 vocab x 4096).
  * `get_transfer_index` is wrapped: it holds the fp64 softmax over the full vocabulary and
    the argmax/scatter index work, timed separately.
  * T_other = T_step - (everything above). It is CPU-bound Python, launch overhead, and the
    device->host syncs -- `(x[:, s:e] == mask_id).sum() == 0` once per step, and
    `replace_position.nonzero(as_tuple=True)` inside the attention cache-write path
    (`model/modeling_llada.py:745,763`), which runs once per layer per pass.

f is reported as T_layers / T_step. Note that T_step is wall clock, so f already accounts
for CPU gaps; that is the f the Stage 0 arithmetic wants (`docs/04` section 1).

Usage (inside an allocation):
    python profile_step.py --mode dual   --n-prompts 20 --out $DLLM_ROOT/results/phase0/profile_A_dual.json
    python profile_step.py --mode prefix --n-prompts 20 --out .../profile_A_prefix.json
    python profile_step.py --mode nocache --n-prompts 4  --out .../profile_A_nocache.json
"""
import argparse, json, os, statistics, sys, time
from collections import defaultdict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate as G                                    # noqa: E402
from transformers import AutoTokenizer                   # noqa: E402
# The fork's own model class, NOT AutoModel: the checkpoint ships its own
# modeling_llada.py whose forward still asserts "The kvcache is not suppotred for MDM"
# (snapshot modeling_llada.py:1218). The fork comments that assert out at
# model/modeling_llada.py:1384, which is why chat.py imports the class directly.
from model.modeling_llada import LLaDAModelLM            # noqa: E402


# ----------------------------------------------------------------------------- prompts
FEWSHOT_TEMPLATE = "Question: {q}\nAnswer: {a}\n\n"

def _gsm8k(split):
    """Read GSM8K straight from the hub parquet: no `datasets` builder, no lock files.

    Two failure modes on this cluster made the builder path unusable inside a GPU job:
    a stale `.incomplete` build directory pinned by an NFS silly-rename (`.nfs*`) file
    failed every stage of job 5696103, and pyarrow sizes its thread pool from
    `hardware_concurrency()` -- 128 on these nodes even inside an 8-CPU allocation --
    which trips `pthread_create` under process-table pressure. Reading the parquet
    directly with threads off avoids both, and needs no network.
    """
    import glob
    import pyarrow
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    import pyarrow.parquet as pq
    root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pat = os.path.join(root, "hub", "datasets--gsm8k", "snapshots", "*", "main",
                       f"{split}-*.parquet")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"no GSM8K {split} parquet under {pat!r} -- "
                                "fetch it with huggingface_hub.snapshot_download('gsm8k', repo_type='dataset')")
    t = pq.read_table(files[0], use_threads=False)
    return t.column("question").to_pylist(), t.column("answer").to_pylist()



def build_prompts(tokenizer, n, n_shot, seed=0):
    """GSM8K test questions with an n-shot prefix from train, then the chat template.

    Matches the shape lm-eval feeds the model in `docs/06` section 2 step 1 closely enough
    for a latency profile; the realised token length is recorded so it can be audited.
    """
    tr_q, tr_a = _gsm8k("train")
    te_q, _ = _gsm8k("test")
    shots = "".join(FEWSHOT_TEMPLATE.format(q=tr_q[i], a=tr_a[i]) for i in range(n_shot))
    out = []
    for i in range(n):
        q = te_q[i]
        user = shots + f"Question: {q}\nAnswer:"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False
        )
        out.append(text)
    return out


# ------------------------------------------------------------------- instrumentation
class PassRecorder:
    """CUDA events + wall clock, one record per model.forward call."""

    def __init__(self, model, block_list):
        self.model, self.blocks = model, block_list
        self.records, self._cur, self._handles = [], None, []
        self._orig_forward = model.forward
        self._orig_gti = G.get_transfer_index
        self._orig_gtid = G.get_transfer_index_dynamic

    # -- event helpers -------------------------------------------------------
    @staticmethod
    def _ev():
        return torch.cuda.Event(enable_timing=True)

    def _block_pre(self, *_):
        if self._cur is None:
            return
        e = self._ev(); e.record(); self._cur["blk_start"].append(e)

    def _block_post(self, *_):
        if self._cur is None:
            return
        e = self._ev(); e.record(); self._cur["blk_end"].append(e)

    # -- wrappers ------------------------------------------------------------
    def _forward(self, *a, **kw):
        now = time.perf_counter()
        rec = {
            "idx": len(self.records),
            "t_start": now,                       # wall clock at the start of THIS pass
            "blk_start": [], "blk_end": [],
            "m_start": self._ev(), "m_end": self._ev(),
            "gti_start": [], "gti_end": [],
            "n_tokens": None, "seq_len": None, "cached": None,
        }
        x = a[0] if a else kw.get("input_ids")
        if torch.is_tensor(x):
            rec["n_tokens"] = int(x.shape[-1])
        pkv = kw.get("past_key_values")
        rec["cached"] = pkv is not None
        rec["seq_len"] = (int(pkv[0][0].shape[2]) + rec["n_tokens"]) if pkv is not None else rec["n_tokens"]
        self._cur = rec
        rec["m_start"].record()
        out = self._orig_forward(*a, **kw)
        rec["m_end"].record()
        self.records.append(rec)
        self._cur = None
        self._pending = rec
        return out

    def _wrap_gti(self, fn):
        def inner(*a, **kw):
            rec = self._pending
            if rec is None:
                return fn(*a, **kw)
            s, e = self._ev(), self._ev()
            s.record(); r = fn(*a, **kw); e.record()
            rec["gti_start"].append(s); rec["gti_end"].append(e)
            return r
        return inner

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self):
        self._pending = None
        for b in self.blocks:
            self._handles.append(b.register_forward_pre_hook(self._block_pre))
            self._handles.append(b.register_forward_hook(self._block_post))
        self.model.forward = self._forward
        G.get_transfer_index = self._wrap_gti(self._orig_gti)
        G.get_transfer_index_dynamic = self._wrap_gti(self._orig_gtid)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.model.forward = self._orig_forward
        G.get_transfer_index = self._orig_gti
        G.get_transfer_index_dynamic = self._orig_gtid

    def mark_step_end(self):
        """Called after each generate() so the last pass's window is closed."""
        self._pending = None

    # -- read-out ------------------------------------------------------------
    def summarise(self):
        torch.cuda.synchronize()
        rows = []
        for r in self.records:
            layers = sum(s.elapsed_time(e) for s, e in zip(r["blk_start"], r["blk_end"]))
            model_gpu = r["m_start"].elapsed_time(r["m_end"])
            gti = sum(s.elapsed_time(e) for s, e in zip(r["gti_start"], r["gti_end"]))
            rows.append(dict(
                idx=r["idx"], n_tokens=r["n_tokens"], seq_len=r["seq_len"], cached=r["cached"],
                n_blocks_seen=len(r["blk_start"]),
                layers_ms=layers, model_gpu_ms=model_gpu, lm_head_ms=model_gpu - layers,
                transfer_ms=gti, step_ms=r["step_ms"],
            ))
        return rows


# ------------------------------------------------------------------------- helpers
def find_blocks(model):
    """Locate the ModuleList of transformer blocks without hard-coding the path."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith("blocks") and len(mod) > 1:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise RuntimeError("could not find the transformer block ModuleList")
    return best


def layer_floor_ms(attn_MB, ffn_MB, kv_MB, n_layers, bw_TBs):
    """docs/04 section 1: the free lower bound on a pass, all layer bytes at peak HBM."""
    return (attn_MB + ffn_MB + kv_MB) * n_layers / (bw_TBs * 1e6) * 1e3


def pct(x, tot):
    return 100.0 * x / tot if tot else float("nan")


# ---------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--mode", choices=["dual", "prefix", "nocache"], default="dual")
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.9, help="tau; pass -1 to disable")
    p.add_argument("--warmup", type=int, default=2, help="prompts discarded before timing")
    p.add_argument("--bw-tbs", type=float, default=4.8, help="peak HBM BW of THIS GPU (H200 NVL ~4.8)")
    p.add_argument("--out", required=True)
    p.add_argument("--trace", default=None, help="also write a torch.profiler trace here (few passes)")
    a = p.parse_args()

    assert torch.cuda.is_available(), "no GPU visible -- this belongs in a SLURM allocation"
    dev = torch.device("cuda")
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(
        a.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(dev).eval()

    blk_name, blocks = find_blocks(model)
    prompts = build_prompts(tok, a.n_prompts + a.warmup, a.n_shot)
    thr = None if a.threshold < 0 else a.threshold

    fn = {"dual": G.generate_with_dual_cache, "prefix": G.generate_with_prefix_cache,
          "nocache": G.generate}[a.mode]

    def run_one(text, rec=None):
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            t0 = time.perf_counter()
            out, nfe = fn(model, ids, steps=a.steps, gen_length=a.gen_length,
                          block_length=a.block_length, temperature=0.0,
                          remasking="low_confidence", threshold=thr)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        return out, nfe, wall, ids.shape[1]

    for t in prompts[:a.warmup]:
        run_one(t)

    per_prompt, all_rows = [], []
    with PassRecorder(model, blocks) as R:
        for t in prompts[a.warmup:]:
            n_before = len(R.records)
            out, nfe, wall, plen = run_one(t)
            R.mark_step_end()
            # close each pass's wall window: end_i - end_{i-1}
            torch.cuda.synchronize()
            new = R.records[n_before:]
            per_prompt.append(dict(nfe=nfe, wall_s=wall, prompt_len=plen,
                                   gen_tokens=int(a.gen_length),
                                   tok_per_s=a.gen_length / wall,
                                   n_passes=len(new)))
            # T_step(i) = start(i+1) - start(i): pass i plus the sampling work after it.
            # The last pass of a prompt has no successor, so it carries no window.
            for i, r in enumerate(new):
                r["step_ms"] = ((new[i + 1]["t_start"] - r["t_start"]) * 1e3
                                if i + 1 < len(new) else float("nan"))
        rows = R.summarise()

    # ------------------------------------------------------------------ aggregate
    # A pass is "cache-writing" when it runs without a past_key_values argument
    # (dual/prefix: the once-per-block full-canvas pass) -- see generate.py.
    groups = defaultdict(list)
    for r in rows:
        if r["step_ms"] != r["step_ms"]:      # NaN: last pass of a prompt, no window
            continue
        if a.mode == "nocache":
            groups["refine"].append(r)        # no cache is ever written; every pass is one
        else:
            groups["cache_write" if not r["cached"] else "refine"].append(r)

    def agg(rs):
        if not rs:
            return None
        med = lambda k: statistics.median(x[k] for x in rs)   # noqa: E731
        step = med("step_ms")
        layers, head, tr = med("layers_ms"), med("lm_head_ms"), med("transfer_ms")
        other = step - (layers + head + tr)
        floor = layer_floor_ms(134.2, 302.0, 16.8, len(blocks), a.bw_tbs)
        return dict(
            n=len(rs), median_tokens=med("n_tokens"), median_seq_len=med("seq_len"),
            step_ms=step, layers_ms=layers, lm_head_ms=head, transfer_ms=tr, other_ms=other,
            layers_pct=pct(layers, step), lm_head_pct=pct(head, step),
            transfer_pct=pct(tr, step), other_pct=pct(other, step),
            f=layers / step if step else float("nan"),
            layer_floor_ms=floor, step_over_floor=step / floor if floor else float("nan"),
        )

    res = dict(
        mode=a.mode, model=a.model, block_module=blk_name, n_layers=len(blocks),
        args=vars(a), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
        driver=getattr(torch.version, "cuda", None),
        per_prompt=per_prompt,
        refine=agg(groups["refine"]), cache_write=agg(groups["cache_write"]),
        rows=rows,
    )
    if res["refine"] and res["cache_write"]:
        res["W"] = res["cache_write"]["step_ms"] / res["refine"]["step_ms"]
    res["N_passes_per_block"] = (
        statistics.median(x["n_passes"] for x in per_prompt) / (a.gen_length / a.block_length)
    )
    res["tokens_per_pass"] = a.gen_length / statistics.median(x["nfe"] for x in per_prompt)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)

    # ------------------------------------------------------------------- stdout
    print(f"\n=== {a.mode}  {a.model}  L={len(blocks)}  GPU={res['gpu']}")
    for label in ("refine", "cache_write"):
        g = res[label]
        if not g:
            print(f"  {label}: no passes"); continue
        print(f"  {label}: n={g['n']}  tokens={g['median_tokens']:.0f}  ctx={g['median_seq_len']:.0f}")
        print(f"     step {g['step_ms']:8.2f} ms   = layers {g['layers_ms']:7.2f} ({g['layers_pct']:5.1f} %)"
              f" + lm_head {g['lm_head_ms']:6.2f} ({g['lm_head_pct']:5.1f} %)"
              f" + transfer {g['transfer_ms']:6.2f} ({g['transfer_pct']:5.1f} %)"
              f" + other {g['other_ms']:6.2f} ({g['other_pct']:5.1f} %)")
        print(f"     f = {g['f']:.3f}   layer floor {g['layer_floor_ms']:.2f} ms"
              f"  -> step is {g['step_over_floor']:.1f}x the floor")
    if "W" in res:
        print(f"  W = T(cache-writing pass) / T(refinement pass) = {res['W']:.2f}")
    print(f"  tokens/pass = {res['tokens_per_pass']:.2f}   passes/block = {res['N_passes_per_block']:.2f}")
    if res["refine"] and "W" in res:
        print(f"  -> Stage 0:  stage0.py --family A --f {res['refine']['f']:.2f} "
              f"--W {res['W']:.1f} --N {res['N_passes_per_block'] - 1:.1f}")
    elif res["refine"]:
        print(f"  (no cache-writing pass in mode {a.mode}: W undefined, "
              f"f = {res['refine']['f']:.3f} is not a deployed-regime number)")
    print(f"  written {a.out}")


if __name__ == "__main__":
    main()
