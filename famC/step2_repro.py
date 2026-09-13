"""Family C step 2 — one reproduction of Nemotron-Labs-TwoTower (`results/familyC/PREREG.md`
step 2 and its 2026-09-12 amendment, which fixes everything below before it ran).

GSM8K 8-shot, 100 test problems (`step2_problems.json`), lm-eval `gsm8k_cot` prompt and scoring.
Two modes of the same checkpoint on the same GPU:
  * `ar`        -- `generate_ar`: the frozen context tower alone, greedy, one token per step
  * `diffusion` -- `generate_mask_diffusion`: gamma 0.8, S 16, T 16 steps per block, greedy

**Time to answer, identically in both modes.** Neither generator stops at a stop string (a base model
does not emit EOS here), and the paper's throughput is "the wall-clock time to produce the final
answer". Both generators call `_extend_context_cache` once per committed token (AR) or block
(diffusion); a wrapper on that call records the committed ids and raises as soon as the text holds
`Q:`, so the clock stops on the same event in both modes. The vendor code is not edited.

Per-problem records are appended to `<out>.jsonl` and a rerun resumes; `<out>.json` is the summary.

    python step2_repro.py --model <snapshot> --problems results/familyC/step2_problems.json \
        --out results/familyC/step2
"""
import argparse
import json
import math
import os
import re
import sys
import time

import torch


class Stop(Exception):
    pass


def build_prompt(spec, question):
    shots = [f"Q: {s['question']}\nA: {s['target']}" for s in spec["fewshot"]]
    return "\n\n".join(shots + [f"Q: {question}\nA:"])


def normalise(s):
    """lm-eval gsm8k_cot exact_match: ignore case, ',', '$', '(?s).*#### ', a trailing '.'."""
    if s is None:
        return None
    s = re.sub(r"(?s).*#### ", "", s)
    s = s.replace(",", "").replace("$", "")
    s = re.sub(r"\.$", "", s.strip())
    return s.lower()


def extract(text):
    m = re.search(r"The answer is (\-?[0-9\.\,]+).", text)
    strict = m.group(1) if m else None
    found = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    flex = None
    if found:
        g = found[-1]
        flex = g[0] or g[1]
    return strict, flex


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True, help="prefix: <out>.jsonl records, <out>.json summary")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps-per-block", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N problems")
    a = ap.parse_args()

    sys.path.insert(0, a.model)          # the modeling files ship inside the snapshot
    from modeling_nemotron_twotower import NemotronHTwoTowerForCausalLM
    from transformers import AutoTokenizer

    spec = json.load(open(a.problems))
    problems = spec["problems"][:a.limit] if a.limit else spec["problems"]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = NemotronHTwoTowerForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                         trust_remote_code=True)
    model = model.cuda().eval()
    dev = next(model.context_tower.parameters()).device
    eos = tok.eos_token_id
    stops = spec["until"]

    # --- the one instrumentation point: the cache-extension call both generators make ----------
    orig_extend = model._extend_context_cache
    orig_den = model._run_denoiser_step_diffusion
    st = {}

    def extend(new_tokens, cache_state, block_wise=True):
        out = orig_extend(new_tokens, cache_state, block_wise=block_wise)
        ids = new_tokens[0].tolist()
        st["ids"].extend(ids)
        st["commits"] += 1
        tail = tok.decode(st["ids"][-(len(ids) + 8):])
        if any(s in tail for s in stops):
            torch.cuda.synchronize()
            st["t_end"] = time.perf_counter()
            raise Stop
        return out

    def denoise(*args, **kw):
        st["nfe"] += 1
        return orig_den(*args, **kw)

    model._extend_context_cache = extend
    model._run_denoiser_step_diffusion = denoise

    def generate(mode, prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        st.update(ids=[], commits=0, nfe=0, t_end=None, stopped=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            try:
                if mode == "ar":
                    out = model.generate_ar(ids, max_new_tokens=a.max_new_tokens, temperature=0.0,
                                            eos_token_id=eos)
                else:
                    out = model.generate_mask_diffusion(
                        ids, max_new_tokens=a.max_new_tokens, block_size=a.block_size,
                        steps_per_block=a.steps_per_block, mask_token_id=3, temperature=0.0,
                        confidence_threshold=a.gamma, eos_token_id=eos)
                torch.cuda.synchronize()
                st["t_end"] = time.perf_counter()
                gen = out[0, ids.shape[1]:].tolist()
            except Stop:
                st["stopped"] = True
                gen = st["ids"]
        text = tok.decode(gen, skip_special_tokens=True)
        cut = min([text.find(s) for s in stops if s in text] or [len(text)])
        return dict(text=text[:cut], wall_s=st["t_end"] - t0, gen_tokens=len(gen),
                    commits=st["commits"], nfe=st["nfe"], prompt_tokens=int(ids.shape[1]),
                    stopped_on_string=st["stopped"])

    # --- warm-up: Triton JIT and autotuning happen on the first calls (fit.json: 26.6 s) --------
    for i in range(a.warmup):
        for mode in ("ar", "diffusion"):
            r = generate(mode, build_prompt(spec, spec["fewshot"][i]["question"]))
            print(f"warm-up {mode} {i}: {r['wall_s']:.2f}s, {r['gen_tokens']} tokens", flush=True)

    done = set()
    rec_path = a.out + ".jsonl"
    if os.path.exists(rec_path):
        for line in open(rec_path):
            r = json.loads(line)
            done.add((r["id"], r["mode"]))
        print(f"resuming: {len(done)} records present", flush=True)
    with open(rec_path, "a") as f:
        for j, p in enumerate(problems):
            order = ("ar", "diffusion") if j % 2 == 0 else ("diffusion", "ar")
            for mode in order:
                if (p["id"], mode) in done:
                    continue
                r = generate(mode, build_prompt(spec, p["question"]))
                strict, flex = extract(r["text"])
                r.update(id=p["id"], mode=mode, gold=p["gold"], pred_strict=strict, pred_flex=flex,
                         correct_strict=normalise(strict) == normalise(p["gold"]),
                         correct_flex=normalise(flex) == normalise(p["gold"]))
                f.write(json.dumps(r) + "\n")
                f.flush()
            if (j + 1) % 10 == 0:
                print(f"  {j+1}/{len(problems)} problems", flush=True)

    recs = [json.loads(l) for l in open(rec_path)]
    ids_all = [p["id"] for p in problems]
    summ = dict(model=a.model, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                n=len(ids_all), args=vars(a), reference=dict(ar_gsm8k=92.49, diffusion_gsm8k=90.14,
                                                             aggregate_throughput_x=2.42))
    by = {}
    for mode in ("ar", "diffusion"):
        rs = {r["id"]: r for r in recs if r["mode"] == mode and r["id"] in ids_all}
        assert len(rs) == len(ids_all), (mode, len(rs))
        k_f = sum(r["correct_flex"] for r in rs.values())
        k_s = sum(r["correct_strict"] for r in rs.values())
        lo, hi = wilson(k_f, len(rs))
        by[mode] = rs
        summ[mode] = dict(acc_flex=k_f / len(rs), acc_strict=k_s / len(rs), wilson95_flex=[lo, hi],
                          total_wall_s=sum(r["wall_s"] for r in rs.values()),
                          mean_gen_tokens=sum(r["gen_tokens"] for r in rs.values()) / len(rs),
                          total_nfe=sum(r["nfe"] for r in rs.values()),
                          stopped_on_string=sum(r["stopped_on_string"] for r in rs.values()))
    ref = summ["reference"]
    summ["throughput_ratio"] = summ["ar"]["total_wall_s"] / summ["diffusion"]["total_wall_s"]
    summ["quality_retained_flex"] = summ["diffusion"]["acc_flex"] / summ["ar"]["acc_flex"]
    both = [i for i in ids_all]
    summ["discordant_flex"] = sum(by["ar"][i]["correct_flex"] != by["diffusion"][i]["correct_flex"] for i in both)
    summ["tolerance"] = dict(
        ar_card_in_ci=summ["ar"]["wilson95_flex"][0] <= ref["ar_gsm8k"] / 100 <= summ["ar"]["wilson95_flex"][1],
        diffusion_card_in_ci=summ["diffusion"]["wilson95_flex"][0] <= ref["diffusion_gsm8k"] / 100
        <= summ["diffusion"]["wilson95_flex"][1],
        diffusion_faster=summ["throughput_ratio"] > 1.0)
    summ["reproduced"] = all(summ["tolerance"].values())
    json.dump(summ, open(a.out + ".json", "w"), indent=1)
    for mode in ("ar", "diffusion"):
        m = summ[mode]
        print(f"{mode:9s} acc flex {m['acc_flex']:.3f} [{m['wilson95_flex'][0]:.3f}, "
              f"{m['wilson95_flex'][1]:.3f}]  strict {m['acc_strict']:.3f}  time {m['total_wall_s']:.1f}s  "
              f"tokens/problem {m['mean_gen_tokens']:.1f}  stopped on string {m['stopped_on_string']}")
    print(f"throughput ratio (AR time / diffusion time) {summ['throughput_ratio']:.3f}  "
          f"(model-card aggregate over generative tasks: 2.42)")
    print(f"quality retained {summ['quality_retained_flex']:.3f}  discordant {summ['discordant_flex']}")
    print("tolerance:", summ["tolerance"], "->", "REPRODUCED" if summ["reproduced"] else "MISS")


if __name__ == "__main__":
    main()
