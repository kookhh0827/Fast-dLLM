"""Family C step 2, baseline fix item 2 (`results/familyC/PREREG.md` 2026-09-13 amendment):
is the low commit rate of step 2 a sampler setting or a property of the model?

30 of step 2's problems, mask diffusion only, S = 16, T in {16, 32} x gamma in {0.8, 0.7}, one model
load. Per block: the denoiser passes it took (NFE) and the masked count entering each pass, so commits
per pass are exact; and whether the block contributes to the extracted answer (starts before the stop
string), which is the population the paper's Fig. 3a uses ("only blocks that contribute to the
extracted answer"). Readings: the NFE-per-block distribution and the share of answer-producing
blocks completing in <= 2 passes. This does not pass or fail step 2; the registered operating point
stays gamma 0.8, S 16.

    python step2_sampler.py --model <snapshot> --problems results/familyC/step2_problems.json \
        --out results/familyC/step2_sampler
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2_repro import Stop, build_prompt, extract, normalise    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--configs", default="16:0.8,32:0.8,16:0.7,32:0.7", help="T:gamma,...")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    a = ap.parse_args()

    sys.path.insert(0, a.model)
    from modeling_nemotron_twotower import NemotronHTwoTowerForCausalLM
    from transformers import AutoTokenizer
    spec = json.load(open(a.problems))
    problems = spec["problems"][:a.n]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = NemotronHTwoTowerForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                         trust_remote_code=True).cuda().eval()
    dev = next(model.context_tower.parameters()).device
    stops = spec["until"]
    orig_extend, orig_den = model._extend_context_cache, model._run_denoiser_step_diffusion
    st = {}

    def extend(new_tokens, cache_state, block_wise=True):
        out = orig_extend(new_tokens, cache_state, block_wise=block_wise)
        st["ids"].extend(new_tokens[0].tolist())
        st["blocks"].append(st["cur"])
        st["cur"] = []
        if any(s in tok.decode(st["ids"][-(new_tokens.shape[1] + 8):]) for s in stops):
            torch.cuda.synchronize()
            st["t_end"] = time.perf_counter()
            raise Stop
        return out

    def denoise(block_ids, *args, **kw):
        st["cur"].append(int((block_ids == 3).sum()))      # masked positions entering this pass
        return orig_den(block_ids, *args, **kw)

    model._extend_context_cache, model._run_denoiser_step_diffusion = extend, denoise

    def run(prompt, T, gamma):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        st.update(ids=[], blocks=[], cur=[], t_end=None)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            try:
                model.generate_mask_diffusion(ids, max_new_tokens=a.max_new_tokens,
                                              block_size=a.block_size, steps_per_block=T,
                                              mask_token_id=3, temperature=0.0,
                                              confidence_threshold=gamma, eos_token_id=tok.eos_token_id)
                torch.cuda.synchronize()
                st["t_end"] = time.perf_counter()
            except Stop:
                pass
        text = tok.decode(st["ids"], skip_special_tokens=True)
        cut = min([text.find(s) for s in stops if s in text] or [len(text)])
        S = a.block_size
        blocks = []
        for b, masked in enumerate(st["blocks"]):
            start_chars = len(tok.decode(st["ids"][:b * S], skip_special_tokens=True))
            commits = [masked[i] - (masked[i + 1] if i + 1 < len(masked) else 0)
                       for i in range(len(masked))]
            blocks.append(dict(nfe=len(masked), commits_per_pass=commits,
                               answer_producing=start_chars < cut))
        return text[:cut], st["t_end"] - t0, blocks

    run(build_prompt(spec, spec["fewshot"][0]["question"]), 16, 0.8)      # warm-up (Triton JIT)
    rec_path = a.out + ".jsonl"
    done = set()
    if os.path.exists(rec_path):
        done = {(r["T"], r["gamma"], r["id"]) for r in map(json.loads, open(rec_path))}
    with open(rec_path, "a") as f:
        for cfg in a.configs.split(","):
            T, gamma = int(cfg.split(":")[0]), float(cfg.split(":")[1])
            for p in problems:
                if (T, gamma, p["id"]) in done:
                    continue
                text, wall, blocks = run(build_prompt(spec, p["question"]), T, gamma)
                _, flex = extract(text)
                f.write(json.dumps(dict(T=T, gamma=gamma, id=p["id"], wall_s=wall, blocks=blocks,
                                        correct_flex=normalise(flex) == normalise(p["gold"]))) + "\n")
                f.flush()
            print(f"  config T={T} gamma={gamma} done", flush=True)

    recs = [json.loads(l) for l in open(rec_path)]
    summ = dict(n=a.n, configs={})
    for cfg in a.configs.split(","):
        T, gamma = int(cfg.split(":")[0]), float(cfg.split(":")[1])
        rs = [r for r in recs if r["T"] == T and r["gamma"] == gamma]
        ab = [b for r in rs for b in r["blocks"] if b["answer_producing"]]
        allb = [b for r in rs for b in r["blocks"]]
        hist = {}
        for b in ab:
            hist[b["nfe"]] = hist.get(b["nfe"], 0) + 1
        first = [b["commits_per_pass"][0] for b in ab if b["commits_per_pass"]]
        summ["configs"][cfg] = dict(
            problems=len(rs), acc_flex=sum(r["correct_flex"] for r in rs) / max(len(rs), 1),
            time_s=sum(r["wall_s"] for r in rs), answer_blocks=len(ab), all_blocks=len(allb),
            nfe_per_answer_block_mean=sum(b["nfe"] for b in ab) / max(len(ab), 1),
            share_answer_blocks_le2=sum(b["nfe"] <= 2 for b in ab) / max(len(ab), 1),
            share_answer_blocks_eq1=sum(b["nfe"] == 1 for b in ab) / max(len(ab), 1),
            tokens_per_nfe_answer_blocks=sum(sum(b["commits_per_pass"]) for b in ab)
            / max(sum(b["nfe"] for b in ab), 1),
            first_pass_commits_mean=sum(first) / max(len(first), 1),
            nfe_hist_answer_blocks=dict(sorted(hist.items())))
    json.dump(summ, open(a.out + ".json", "w"), indent=1)
    for cfg, c in summ["configs"].items():
        print(f"T:gamma {cfg:7s} acc {c['acc_flex']:.3f}  time {c['time_s']:.0f}s  "
              f"NFE/answer-block {c['nfe_per_answer_block_mean']:.2f}  <=2 passes {c['share_answer_blocks_le2']:.3f}  "
              f"1 pass {c['share_answer_blocks_eq1']:.3f}  tok/NFE {c['tokens_per_nfe_answer_blocks']:.2f}  "
              f"first-pass commits {c['first_pass_commits_mean']:.2f}")
        print(f"   NFE histogram (answer-producing blocks): {c['nfe_hist_answer_blocks']}")


if __name__ == "__main__":
    main()
