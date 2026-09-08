"""Phase 0 step 6 -- batch-1 throughput of the AR parent, Qwen2.5-7B-Instruct.

`docs/06` section 2 step 6 wants this row in every Family B table: HF `generate` with a KV
cache, greedy, batch 1, the same prompts and max_new_tokens as the dLLM rows, same GPU.
lm-eval's `hf` model reports accuracy but not tokens/second, so the throughput half of the
row is measured here. The literature anchor is 39.5 tok/s at batch 1 on an A100
(`docs/papers/fast_dllm_v2.md` Fig. 1b) -- an anchor, never a row.

Every number this prints is batch 1 in the memory-bound regime, which `docs/04` says is the
only regime where a dLLM stack's advantage over AR exists at all.
"""
import argparse, json, os, statistics, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_step_v2 as P                                    # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--n-prompts", type=int, default=20)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--max-new-tokens", type=int, default=256)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()

# The same prompts the Family B rows use, so the comparison is like for like.
prompts = P.build_prompts(tok, a.n_prompts + a.warmup)

def run(text):
    ids = tok([text], return_tensors="pt").to(dev)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n_new = out.shape[1] - ids["input_ids"].shape[1]
    return n_new, dt

for t in prompts[:a.warmup]:
    run(t)

rows = []
for t in prompts[a.warmup:]:
    n, dt = run(t)
    rows.append(dict(new_tokens=int(n), seconds=dt, tok_per_s=n / dt))
    print(f"  {n:4d} tokens in {dt:6.2f} s -> {n/dt:7.2f} tok/s", flush=True)

tps = [r["tok_per_s"] for r in rows]
res = dict(model=a.model, gpu=torch.cuda.get_device_name(0), args=vars(a), rows=rows,
           median_tok_per_s=statistics.median(tps),
           mean_tok_per_s=sum(tps) / len(tps),
           note="batch 1, memory-bound; HF generate with KV cache, greedy")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print(f"\n  median {res['median_tok_per_s']:.2f} tok/s  (batch 1, {res['gpu']})")
print(f"  A100 anchor from the paper: 39.5 tok/s at batch 1 -- an anchor, not a row")
print(f"  written {a.out}")
