"""Family B output sanity: decode what the profiled gate configuration actually generates.

The Phase 0 profile never looked at a single token. N and W are read off the realised
trajectory, so a broken sampler would corrupt the Stage 0 inputs while the timing tables
stayed healthy.
"""
import argparse, json, os, re, sys, types
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generation_functions                                    # noqa: E402
import profile_step_v2 as P                                    # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM   # noqa: E402


def gold_answer(a):
    m = re.search(r"####\s*([-\d,\.]+)", a)
    return m.group(1).replace(",", "").strip() if m else None


def pred_answer(t):
    m = re.findall(r"boxed\{([^}]*)\}", t)
    src = m[-1] if m else t
    nums = re.findall(r"-?\d[\d,]*\.?\d*", src.replace(",", ""))
    return nums[-1].strip() if nums else None


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
ap.add_argument("--n-prompts", type=int, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True,
                                             torch_dtype=torch.bfloat16).to(dev).eval()
model.mdm_sample = types.MethodType(
    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model)

prompts = P.build_prompts(tok, a.n_prompts)
_, answers = P._gsm8k("test")
out_rows = []
for i, text in enumerate(prompts):
    ids = tok([text], return_tensors="pt").input_ids.to(dev)
    L = ids.shape[1]
    with torch.no_grad():
        o = model.mdm_sample(ids, tokenizer=tok, block_size=32, small_block_size=32,
                             max_new_tokens=256, mask_id=151665, min_len=L,
                             seq_len=torch.tensor([L], device=dev),
                             use_block_cache=False, threshold=0.9)
    gen = tok.decode(o[0][L:], skip_special_tokens=True)
    g, q = gold_answer(answers[i]), pred_answer(gen)
    ok = (g is not None and q is not None and abs(float(q) - float(g)) < 1e-6)
    out_rows.append(dict(idx=i, gold=g, pred=q, correct=bool(ok), text=gen))
    print(f"\n--- problem {i}  gold={g}  pred={q}  {'CORRECT' if ok else 'WRONG'}")
    print("   " + gen.strip().replace("\n", "\n   ")[:600])
print(f"\n  -> {sum(r['correct'] for r in out_rows)} / {len(out_rows)} correct (sanity, not accuracy)")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(dict(model=a.model, rows=out_rows), open(a.out, "w"), indent=2)
print("  written", a.out)
