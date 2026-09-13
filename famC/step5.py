"""Family C step 5 on Nemotron-Labs-TwoTower (`results/familyC/PREREG.md` 2026-09-13 (c)).

One allocation, resumable per (cell, problem):
  P. pair-level cosine -- each MoE sublayer with the non-MoE sublayers before it as one unit (the
     residual updates between two MoE outputs), cos(stream before the unit, stream after its MoE) on
     the block's masked positions, by r-bucket, on the 128 calibration prompts; the unit that matches a
     LLaDA/Qwen layer (one mixer + one FFN) for the note's A/B comparison;
  C. cells on the 100 step-2 GSM8K test problems, greedy, S 16, T 16, time to answer:
     full@0.8 (replicate 1) -> identity on k in {3,5,7} MoE sublayers @0.8 -> full@0.7 -> k3/5/7 @0.7
     -> full@0.6 -> AR -> full@0.8 (replicate 2, last, so the pair also brackets drift).
     MoE selection: step 3's per-sublayer cosine ranking over the 23 MoE sublayers, greedy, no two
     consecutive in MoE order, hybrid sublayers 0 and 51 protected. The skip applies to every denoiser
     pass; the context tower always runs at full depth (it is the cache writer). No KL-greedy cell:
     porting the teacher-forced search to this model exceeds the half-day the PREREG allows.
Per problem: accuracy, wall-clock to answer, NFE, and per block the masked count entering every pass
(so commits per pass and the block's r at each pass are exact).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step2_repro import Stop, build_prompt, extract, normalise    # noqa: E402
import denoiser_loop as DL                                         # noqa: E402

NB = 4


def select_moe(types, cos, k):
    moe = [i for i, t in enumerate(types) if t == "moe" and i not in (0, len(types) - 1)]
    pos = {l: j for j, l in enumerate([i for i, t in enumerate(types) if t == "moe"])}
    chosen = []
    for l in sorted(moe, key=lambda l: -cos[l]):
        if len(chosen) == k:
            break
        if any(abs(pos[l] - pos[c]) == 1 for c in chosen):
            continue
        chosen.append(l)
    assert len(chosen) == k, (k, chosen)
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--step34", required=True)
    ap.add_argument("--out", required=True, help="prefix: <out>_pairs.json, <out>.jsonl, <out>_cells.json")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: problems per cell / calib prompts")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()

    sys.path.insert(0, a.model)
    from modeling_nemotron_twotower import NemotronHTwoTowerForCausalLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = NemotronHTwoTowerForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                         trust_remote_code=True).cuda().eval()
    dev = next(model.context_tower.parameters()).device
    s34 = json.load(open(a.step34))
    types = s34["layer_types"]
    cos = s34["calibration"]["global_cosine"]
    leq = {r["layer"]: r["L_eq"] for r in s34["bytes"]["per_layer"]}
    spec = json.load(open(a.problems))
    cal = json.load(open(a.calib))
    problems = spec["problems"][:a.limit] if a.limit else spec["problems"]
    stops = spec["until"]

    st = {"skip": set()}
    orig_extend = model._extend_context_cache

    def extend(new_tokens, cache_state, block_wise=True):
        out = orig_extend(new_tokens, cache_state, block_wise=block_wise)
        st["ids"].extend(new_tokens[0].tolist())
        st["blocks"].append(st["cur"]); st["cur"] = []
        if any(s in tok.decode(st["ids"][-(new_tokens.shape[1] + 8):]) for s in stops):
            torch.cuda.synchronize(); st["t_end"] = time.perf_counter()
            raise Stop
        return out
    model._extend_context_cache = extend

    probe_state = {}

    def run_probe(li, btype, h_in, h_out, den_input):
        if probe_state.get("on") is None:
            return
        m = probe_state["mask"]
        if li == 0 or types[li - 1] == "moe":
            probe_state["unit_in"] = h_in
        if btype == "moe" and m.any():
            a_ = probe_state["unit_in"][0][m].float(); b_ = h_out[0][m].float()
            probe_state["sum"][probe_state["b"]][li] += float(F.cosine_similarity(a_, b_, dim=-1).mean())
            probe_state["cnt"][probe_state["b"]][li] += 1

    def install(skip, probe):
        DL.install(model, probe=run_probe if probe else None, skip=skip)
        inner = model._run_denoiser_step_diffusion

        def call(block_ids, cache_state, t=None, den_cache=None):
            masked = int((block_ids == 3).sum())
            st["cur"].append(masked)
            if probe:
                m = (block_ids.to(dev) == 3)[0]
                probe_state.update(on=True, mask=m, b=min(int((1 - float(m.float().mean())) * NB), NB - 1))
            return inner(block_ids, cache_state, t=t, den_cache=den_cache)
        model._run_denoiser_step_diffusion = call

    def generate(question, mode, gamma, spec_):
        ids = tok(build_prompt(spec_, question), return_tensors="pt").input_ids.to(dev)
        st.update(ids=[], blocks=[], cur=[], t_end=None)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            try:
                if mode == "ar":
                    out = model.generate_ar(ids, max_new_tokens=a.max_new_tokens, temperature=0.0,
                                            eos_token_id=tok.eos_token_id)
                else:
                    out = model.generate_mask_diffusion(ids, max_new_tokens=a.max_new_tokens, block_size=16,
                                                        steps_per_block=16, mask_token_id=3, temperature=0.0,
                                                        confidence_threshold=gamma, eos_token_id=tok.eos_token_id)
                torch.cuda.synchronize(); st["t_end"] = time.perf_counter()
            except Stop:
                pass
        text = tok.decode(st["ids"], skip_special_tokens=True)
        cut = min([text.find(s) for s in stops if s in text] or [len(text)])
        return text[:cut], st["t_end"] - t0

    # ---- P. pair-level cosine -------------------------------------------------------------------------
    pairs_path = a.out + "_pairs.json"
    if not os.path.exists(pairs_path):
        L = len(types)
        probe_state.update(sum=[[0.0] * L for _ in range(NB)], cnt=[[0] * L for _ in range(NB)])
        install(set(), probe=True)
        generate(cal["fewshot"][0]["question"], "diffusion", 0.8, cal)            # warm-up
        probe_state.update(sum=[[0.0] * L for _ in range(NB)], cnt=[[0] * L for _ in range(NB)])
        prompts = cal["problems"][:a.limit] if a.limit else cal["problems"]
        for j, p in enumerate(prompts):
            generate(p["question"], "diffusion", 0.8, cal)
            if (j + 1) % 32 == 0:
                print(f"  pairs {j+1}/{len(prompts)}", flush=True)
        units, start = [], 0
        for i, t in enumerate(types):
            if t == "moe":
                units.append(dict(sublayers=list(range(start, i + 1)),
                                  kinds="".join({"mamba": "M", "attention": "*", "moe": "E"}[types[x]]
                                                for x in range(start, i + 1))))
                start = i + 1
        for u in units:
            e = u["sublayers"][-1]
            byb = [probe_state["sum"][b][e] / probe_state["cnt"][b][e] if probe_state["cnt"][b][e] else None
                   for b in range(NB)]
            u.update(by_bucket=byb, cosine=sum(x for x in byb if x is not None) / max(sum(x is not None for x in byb), 1))
        runs, run = [], []
        for j, u in enumerate(units):
            if u["cosine"] > 0.95:
                run.append(j)
            else:
                if run: runs.append(run)
                run = []
        if run: runs.append(run)
        json.dump(dict(units=units, n_units=len(units), runs_above_095=runs,
                       longest_run_above_095=max((len(r) for r in runs), default=0),
                       buckets=[f"b{b}: r in ({1-(b+1)/NB:.2f}, {1-b/NB:.2f}]" for b in range(NB)],
                       note="unit = the sublayers after the previous MoE through this MoE; cos(stream before the "
                            "unit, stream after it) on masked positions, 128 calibration prompts, gamma 0.8"),
                  open(pairs_path, "w"), indent=1)
        print("pairs:", [(u["kinds"], round(u["cosine"], 4)) for u in units], flush=True)
        probe_state["on"] = None

    # ---- C. cells -----------------------------------------------------------------------------------------
    sets = {k: select_moe(types, cos, k) for k in (3, 5, 7)}
    cells = [("full_g0.8_rep1", "diffusion", 0.8, []), ("k3_g0.8", "diffusion", 0.8, sets[3]),
             ("k5_g0.8", "diffusion", 0.8, sets[5]), ("k7_g0.8", "diffusion", 0.8, sets[7]),
             ("full_g0.7", "diffusion", 0.7, []), ("k3_g0.7", "diffusion", 0.7, sets[3]),
             ("k5_g0.7", "diffusion", 0.7, sets[5]), ("k7_g0.7", "diffusion", 0.7, sets[7]),
             ("full_g0.6", "diffusion", 0.6, []), ("ar", "ar", None, []),
             ("full_g0.8_rep2", "diffusion", 0.8, [])]
    json.dump(dict(cells=[dict(name=c[0], mode=c[1], gamma=c[2], skip=c[3],
                               L_eq_per_pass=sum(leq[l] for l in c[3])) for c in cells],
                   moe_order=[i for i, t in enumerate(types) if t == "moe"],
                   selection="step-3 cosine ranking over MoE sublayers, no two consecutive in MoE order, "
                             "sublayers 0 and 51 protected"),
              open(a.out + "_cells.json", "w"), indent=1)
    print("skip sets:", sets, flush=True)
    rec_path = a.out + ".jsonl"
    done = set()
    if os.path.exists(rec_path):
        done = {(r["cell"], r["id"]) for r in map(json.loads, open(rec_path))}
    warmed = False
    with open(rec_path, "a") as f:
        for name, mode, gamma, skip in cells:
            if all((name, p["id"]) in done for p in problems):
                continue
            install(set(skip), probe=False)
            if not warmed:
                generate(spec["fewshot"][0]["question"], "diffusion", 0.8, spec); warmed = True
            t0 = time.perf_counter()
            for p in problems:
                if (name, p["id"]) in done:
                    continue
                text, wall = generate(p["question"], mode, gamma, spec)
                _, flex = extract(text)
                f.write(json.dumps(dict(cell=name, id=p["id"], wall_s=wall, gen_tokens=len(st["ids"]),
                                        nfe=sum(len(b) for b in st["blocks"]) if mode != "ar" else None,
                                        blocks=st["blocks"] if mode != "ar" else None,
                                        pred_flex=flex, gold=p["gold"],
                                        correct_flex=normalise(flex) == normalise(p["gold"]))) + "\n")
                f.flush()
            print(f"  cell {name} done ({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
