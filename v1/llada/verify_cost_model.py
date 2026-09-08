"""Two checks on the Phase 0 profile, both of which can invalidate its GO verdict.

CHECK 1 -- output sanity. The profile never decoded anything. Garbage generation would
produce the same timing tables, and N and W are read off the realised trajectory, so a
broken sampler would corrupt the Stage 0 inputs while looking perfectly healthy.

CHECK 2 -- depth linearity. Stage 0 assumes that removing a fraction phi of the per-pass
layer bytes removes phi*f of the step. The profile measured f as a *share*, never that the
share is proportional to depth. If anything inside the block loop is a fixed per-pass cost,
the real saving is smaller than the model and L_eq_req is understated. This times a
refinement pass with k layers replaced by an identity that forwards the existing cache
(the shape of the `identity` mode), and compares against 1 - f*(k/L).

    python verify_cost_model.py --n-prompts 3 --ks 0,4,8,12,16 --out <json>
"""
import argparse, json, os, re, statistics, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate as G                                          # noqa: E402
import profile_step as P                                      # noqa: E402
from transformers import AutoTokenizer                        # noqa: E402
from model.modeling_llada import LLaDAModelLM                 # noqa: E402


def make_identity(orig):
    """A skipped block: pass the residual through and hand the cache back untouched.

    It skips only when a cache already exists, i.e. on refinement passes. The
    cache-writing pass at each block start runs at FULL depth -- both because
    docs/06 section 2.5 Stage 2 requires it in every cell, and because a skipped block
    on that pass has no K/V to return and trips `assert cache is not None`
    (model/modeling_llada.py:1496).
    """
    def f(x, attention_bias=None, layer_past=None, use_cache=False, replace_position=None):
        if layer_past is None:
            return orig(x, attention_bias=attention_bias, layer_past=layer_past,
                        use_cache=use_cache, replace_position=replace_position)
        return x, layer_past
    return f


def skip_set(L, k, keep_last=8):
    """Non-adjacent, first and last protected; keep_last relaxed only when k needs it.

    docs/06 section 2.5 prescribes relaxing keep_last rather than giving up when the
    requirement exceeds its ceiling (12 of 32 here), so the probe does the same and
    reports which rule it used.
    """
    for kl in (keep_last, 0):
        pool = list(range(1, L - kl if kl else L - 1))
        chosen = pool[::2][:k]
        if len(chosen) == k:
            return chosen
    raise AssertionError(f"cannot place {k} non-adjacent layers in L={L}")


def gold_answer(a):
    m = re.search(r"####\s*([-\d,\.]+)", a)
    return m.group(1).replace(",", "").strip() if m else None


def pred_answer(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1].strip() if nums else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--n-prompts", type=int, default=3)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--ks", default="0,4,8,12,16")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    blk_name, blocks = P.find_blocks(model)
    L = len(blocks)
    prompts = P.build_prompts(tok, a.n_prompts + 1, a.n_shot)
    _, gold_q_a = P._gsm8k("test")

    res = {"model": a.model, "L": L, "gpu": torch.cuda.get_device_name(0), "args": vars(a)}

    # ---------------------------------------------------------------- CHECK 1
    print("=== CHECK 1: decode what the profiled configuration actually generates")
    samples = []
    for i, text in enumerate(prompts[1:]):                    # prompts[0] is warm-up
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out, nfe = G.generate_with_dual_cache(
                model, ids, steps=a.gen_length, gen_length=a.gen_length,
                block_length=a.block_length, temperature=0.0,
                remasking="low_confidence", threshold=a.threshold)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        g = gold_answer(gold_q_a[i + 1]); q = pred_answer(gen)
        ok = (g is not None and q is not None and abs(float(q) - float(g)) < 1e-6)
        samples.append(dict(idx=i + 1, nfe=nfe, gold=g, pred=q, correct=bool(ok), text=gen))
        print(f"\n--- problem {i}  nfe={nfe}  gold={g}  pred={q}  {'CORRECT' if ok else 'WRONG'}")
        print("   " + gen.strip().replace("\n", "\n   ")[:600])
    res["check1_samples"] = samples
    res["check1_correct"] = sum(s["correct"] for s in samples)
    print(f"\n  -> {res['check1_correct']} / {len(samples)} correct "
          f"(3 problems is a sanity check, not an accuracy measurement)")

    # ---------------------------------------------------------------- CHECK 2
    print("\n=== CHECK 2: does a refinement pass scale with depth?")
    ks = [int(x) for x in a.ks.split(",")]
    scaling, base_step, base_f = [], None, None
    for k in ks:
        chosen = skip_set(L, k)
        originals = {}
        for j in chosen:
            originals[j] = blocks[j].forward
            blocks[j].forward = make_identity(originals[j])
        try:
            with P.PassRecorder(model, blocks) as R:
                for text in prompts[:1 + a.n_prompts]:        # first is warm-up for this k
                    ids = tok(text, return_tensors="pt").input_ids.to(dev)
                    n0 = len(R.records)
                    with torch.no_grad():
                        G.generate_with_dual_cache(
                            model, ids, steps=a.gen_length, gen_length=a.gen_length,
                            block_length=a.block_length, temperature=0.0,
                            remasking="low_confidence", threshold=a.threshold)
                        torch.cuda.synchronize()
                    new = R.records[n0:]
                    for i2, r in enumerate(new):
                        r["step_ms"] = ((new[i2 + 1]["t_start"] - r["t_start"]) * 1e3
                                        if i2 + 1 < len(new) else float("nan"))
                rows = R.summarise()
        finally:
            for j, f in originals.items():
                blocks[j].forward = f
        ref = [r for r in rows if r["step_ms"] == r["step_ms"] and r["n_tokens"] == a.block_length]
        step = statistics.median(r["step_ms"] for r in ref)
        lay = statistics.median(r["layers_ms"] for r in ref)
        nfe = len(rows)
        if k == 0:
            base_step, base_f = step, lay / step
        pred = 1.0 - base_f * (k / L)
        scaling.append(dict(k=k, skipped=chosen, n=len(ref), step_ms=step, layers_ms=lay,
                            measured_ratio=step / base_step, predicted_ratio=pred,
                            n_passes=nfe))
        print(f"  k={k:3d}  step {step:7.3f} ms  layers {lay:7.3f}  "
              f"measured T(k)/T(0) = {step/base_step:.4f}   model 1-f*k/L = {pred:.4f}   "
              f"delta = {100*(step/base_step - pred):+5.2f} pp   passes={nfe}")
    res["check2_scaling"] = scaling
    res["check2_base_f"] = base_f

    err = [abs(s["measured_ratio"] - s["predicted_ratio"]) for s in scaling]
    res["check2_max_abs_error_pp"] = 100 * max(err)
    print(f"\n  -> worst deviation from the cost model: {100*max(err):.2f} percentage points")
    print("     (the model is optimistic if measured > predicted)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"  written {a.out}")


if __name__ == "__main__":
    main()
