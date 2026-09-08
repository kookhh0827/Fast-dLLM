"""Phase 0 step 7 / runbook section 0a step 1 -- per-pass latency split for Fast-dLLM v2 7B.

Family B counterpart of `v1/llada/profile_step.py`. Same question, different pass taxonomy.
`v2/generation_functions.py:batch_sample` issues three kinds of forward:

  prefill        line 37   update_past_key_values=True,  past_key_values=None   (once per generation)
  encode         line 81   update_past_key_values=True,  past_key_values=...    (once per block)  -> W
  refinement     line 111  update_past_key_values=False                          (the gate setting)
  sub-block      lines 103/108, only when use_block_cache=True -- not the gate setting

so W = T(encode) / T(refinement), which `docs/06` section 2.5 predicts is about 1, and the
prefill is recorded separately (it is amortised over all blocks and sits outside W).

The sampling work in Family B is inline in `batch_sample` rather than in a named function,
so the split is the coarser {layers / head+embed+norm / post-forward}; post-forward holds
the confidence softmax, the argmax/scatter and the Python and syncs. That is enough for the
section 0a step 2 question, which turns only on f = T_layers / T_step. Pass --trace for a
torch.profiler trace when the post-forward bucket needs breaking down by kernel.

Usage (inside an allocation):
    python profile_step_v2.py --n-prompts 20 --out $DLLM_ROOT/results/phase0/profile_B_gate.json
"""
import argparse, json, os, statistics, sys, time, types
from collections import defaultdict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generation_functions                                        # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM       # noqa: E402

GSM8K_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{{}}."

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



def build_prompts(tokenizer, n):
    """0-shot GSM8K in the shape `v2/eval.py` builds it (eval.py:237-243)."""
    te_q, _ = _gsm8k("test")
    out = []
    for i in range(n):
        q = f"Question: {te_q[i]}\nAnswer:".replace("Answer:", GSM8K_INSTRUCTION)
        out.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False))
    return out


class PassRecorder:
    def __init__(self, model, blocks, head):
        self.model, self.blocks, self.head = model, blocks, head
        self.records, self._cur, self._handles = [], None, []
        self._orig_forward = model.forward

    @staticmethod
    def _ev():
        return torch.cuda.Event(enable_timing=True)

    def _mk(self, key):
        def hook(*_):
            if self._cur is not None:
                e = self._ev(); e.record(); self._cur[key].append(e)
        return hook

    def _forward(self, *a, **kw):
        upkv = bool(kw.get("update_past_key_values", False))
        pkv = kw.get("past_key_values")
        x = kw.get("input_ids", a[0] if a else None)
        rec = dict(
            idx=len(self.records), t_start=time.perf_counter(),
            blk_start=[], blk_end=[], head_start=[], head_end=[],
            m_start=self._ev(), m_end=self._ev(),
            n_tokens=int(x.shape[-1]) if torch.is_tensor(x) else None,
            update_kv=upkv, has_kv=pkv is not None,
            kind=("prefill" if upkv and pkv is None else "encode" if upkv else "refine"),
            use_block_cache=bool(kw.get("use_block_cache", False)),
        )
        self._cur = rec
        rec["m_start"].record()
        out = self._orig_forward(*a, **kw)
        rec["m_end"].record()
        self.records.append(rec)
        self._cur = None
        return out

    def __enter__(self):
        for b in self.blocks:
            self._handles.append(b.register_forward_pre_hook(self._mk("blk_start")))
            self._handles.append(b.register_forward_hook(self._mk("blk_end")))
        if self.head is not None:
            self._handles.append(self.head.register_forward_pre_hook(self._mk("head_start")))
            self._handles.append(self.head.register_forward_hook(self._mk("head_end")))
        self.model.forward = self._forward
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.model.forward = self._orig_forward

    def summarise(self):
        torch.cuda.synchronize()
        rows = []
        for r in self.records:
            layers = sum(s.elapsed_time(e) for s, e in zip(r["blk_start"], r["blk_end"]))
            head = sum(s.elapsed_time(e) for s, e in zip(r["head_start"], r["head_end"]))
            gpu = r["m_start"].elapsed_time(r["m_end"])
            rows.append(dict(
                idx=r["idx"], kind=r["kind"], n_tokens=r["n_tokens"],
                n_blocks_seen=len(r["blk_start"]),
                layers_ms=layers, lm_head_ms=head, model_gpu_ms=gpu,
                rest_in_fwd_ms=gpu - layers - head, step_ms=r["step_ms"],
            ))
        return rows


def find_module_list(model, suffix="layers"):
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith(suffix) and len(mod) > 1:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise RuntimeError(f"could not find a ModuleList ending in {suffix!r}")
    return best


def layer_floor_ms(attn_MB, ffn_MB, kv_MB, n_layers, bw_TBs):
    return (attn_MB + ffn_MB + kv_MB) * n_layers / (bw_TBs * 1e6) * 1e3


def pct(x, tot):
    return 100.0 * x / tot if tot else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--small-block-size", type=int, default=32, help="32 = the gate setting")
    p.add_argument("--use-block-cache", action="store_true", help="sub-block 8 path; for the record only")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--mask-id", type=int, default=151665)
    p.add_argument("--bw-tbs", type=float, default=4.8)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    assert torch.cuda.is_available(), "no GPU visible -- this belongs in a SLURM allocation"
    dev = torch.device("cuda")
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    # exactly what v2/eval.py:86 does
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model)

    blk_name, blocks = find_module_list(model, "layers")
    head = getattr(model, "lm_head", None)
    prompts = build_prompts(tok, a.n_prompts + a.warmup)

    def run_one(text):
        # No padding: v2/eval.py pads only up to the batch maximum, which at batch 1
        # is the prompt itself. Padding to a multiple of block_size is NOT what the
        # reference does and desynchronises min_len/seq_len from input_ids.
        ids = tok([text], return_tensors="pt").input_ids.to(dev)
        L = ids.shape[1]
        with torch.no_grad():
            t0 = time.perf_counter()
            out = model.mdm_sample(
                ids, tokenizer=tok, block_size=a.block_size,
                small_block_size=a.small_block_size, max_new_tokens=a.max_new_tokens,
                mask_id=a.mask_id, min_len=L, seq_len=torch.tensor([L], device=dev),
                use_block_cache=a.use_block_cache, threshold=a.threshold)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        return out, wall, L

    for t in prompts[:a.warmup]:
        run_one(t)

    per_prompt = []
    with PassRecorder(model, blocks, head) as R:
        for t in prompts[a.warmup:]:
            n_before = len(R.records)
            out, wall, plen = run_one(t)
            new = R.records[n_before:]
            for i, r in enumerate(new):
                r["step_ms"] = ((new[i + 1]["t_start"] - r["t_start"]) * 1e3
                                if i + 1 < len(new) else float("nan"))
            n_ref = sum(1 for r in new if r["kind"] == "refine")
            per_prompt.append(dict(wall_s=wall, prompt_len=plen, n_passes=len(new),
                                   n_refine=n_ref, n_encode=sum(1 for r in new if r["kind"] == "encode"),
                                   tok_per_s=a.max_new_tokens / wall))
        rows = R.summarise()

    groups = defaultdict(list)
    for r in rows:
        if r["step_ms"] == r["step_ms"]:      # drop the NaN tail pass of each prompt
            groups[r["kind"]].append(r)

    def agg(rs):
        if not rs:
            return None
        med = lambda k: statistics.median(x[k] for x in rs)          # noqa: E731
        step, layers, hd = med("step_ms"), med("layers_ms"), med("lm_head_ms")
        rest = med("rest_in_fwd_ms")
        post = step - (layers + hd + rest)
        floor = layer_floor_ms(58.7, 407.4, 2.1, len(blocks), a.bw_tbs)
        return dict(n=len(rs), median_tokens=med("n_tokens"), step_ms=step,
                    layers_ms=layers, lm_head_ms=hd, rest_in_fwd_ms=rest, post_forward_ms=post,
                    layers_pct=pct(layers, step), lm_head_pct=pct(hd, step),
                    post_forward_pct=pct(post + rest, step),
                    f=layers / step if step else float("nan"),
                    layer_floor_ms=floor, step_over_floor=step / floor if floor else float("nan"))

    res = dict(model=a.model, block_module=blk_name, n_layers=len(blocks), args=vars(a),
               torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
               per_prompt=per_prompt, rows=rows,
               refine=agg(groups["refine"]), encode=agg(groups["encode"]), prefill=agg(groups["prefill"]))
    if res["refine"] and res["encode"]:
        res["W"] = res["encode"]["step_ms"] / res["refine"]["step_ms"]
    n_blocks = a.max_new_tokens / a.block_size
    res["N_passes_per_block"] = statistics.median(x["n_refine"] for x in per_prompt) / n_blocks
    res["tokens_per_pass"] = a.max_new_tokens / statistics.median(
        x["n_refine"] + x["n_encode"] for x in per_prompt)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    print(f"\n=== Fast-dLLM v2  L={len(blocks)}  GPU={res['gpu']}  "
          f"small_block={a.small_block_size} block_cache={a.use_block_cache} tau={a.threshold}")
    for label in ("refine", "encode", "prefill"):
        g = res[label]
        if not g:
            print(f"  {label}: no passes"); continue
        print(f"  {label:9s} n={g['n']:4d} tokens={g['median_tokens']:.0f}  step {g['step_ms']:8.2f} ms"
              f" = layers {g['layers_ms']:7.2f} ({g['layers_pct']:5.1f} %)"
              f" + lm_head {g['lm_head_ms']:6.2f} ({g['lm_head_pct']:5.1f} %)"
              f" + post/other {g['post_forward_ms'] + g['rest_in_fwd_ms']:6.2f} ({g['post_forward_pct']:5.1f} %)")
        print(f"            f = {g['f']:.3f}   layer floor {g['layer_floor_ms']:.2f} ms"
              f"  -> {g['step_over_floor']:.1f}x the floor")
    if "W" in res:
        print(f"  W = T(encode)/T(refine) = {res['W']:.2f}   (docs/06 section 2.5 predicts ~1)")
    print(f"  refinement passes per block N = {res['N_passes_per_block']:.2f}   "
          f"tokens/pass = {res['tokens_per_pass']:.2f}")
    print(f"  -> Stage 0:  stage0.py --family B --f {res['refine']['f']:.2f} "
          f"--W {res.get('W', float('nan')):.1f} --N {res['N_passes_per_block']:.1f}")
    print(f"  written {a.out}")


if __name__ == "__main__":
    main()
