"""Family C steps 3 and 4 on Nemotron-Labs-TwoTower (`results/familyC/PREREG.md` steps 3-4 as
amended 2026-09-13 (b)). One model load, four phases:

  0. null check -- the instrumented loop copy (`denoiser_loop.py`) must give bitwise the vendor
     method's logits on an all-mask and a half-mask block, or nothing runs;
  1. timing (Stage 0's f, W, N) -- warmed-up mask diffusion (gamma 0.8, S 16, T 16) on 8 of the
     calibration prompts, no probes, CUDA-synchronised: per pass the layer loop and the whole step,
     per block the context-tower extension and the denoiser-cache build;
  2. calibration (step 3) -- cos(h_in, h_out) of every denoiser layer on the block's masked positions,
     by layer type and r-bucket (r = masked fraction entering the pass, the model's own time input;
     4 buckets as in Families A/B), on the 128 GSM8K-train calibration prompts shared with A and B;
     the same passes record the routed experts per MoE layer (unique over the 16-token pass) and the
     context length;
  3. bytes -- per layer type, from the tensors themselves: mixer weights, the MoE router and shared
     expert, the measured mean active experts x each expert's bytes, and the state each layer reads
     per pass (Mamba conv + SSM state; attention KV at the measured context).

Outputs `<out>.json` (everything) and prints the tables. Stage 0 itself is `scripts/analysis/stage0.py`,
run on these numbers afterwards -- never by hand.
"""
import argparse
import json
import os
import sys
import time
import types
from collections import defaultdict

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step2_repro import Stop, build_prompt                          # noqa: E402
import denoiser_loop as DL                                           # noqa: E402

NB = 4


def bucket_of(r):
    return min(int((1.0 - r) * NB), NB - 1)          # b0 = r in (0.75, 1], as calibrate_depth*.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="step3_calib_prompts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--n-timing", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()

    sys.path.insert(0, a.model)
    from modeling_nemotron_twotower import NemotronHTwoTowerForCausalLM
    from transformers import AutoTokenizer
    spec = json.load(open(a.prompts))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = NemotronHTwoTowerForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                         trust_remote_code=True).cuda().eval()
    dev = next(model.context_tower.parameters()).device
    layers = model.denoiser_tower.layers
    types_ = [b.block_type for b in layers]
    L = len(layers)
    stops = spec["until"]
    res = dict(model=a.model, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               args=vars(a), layer_types=types_)

    # ---- 0. null check ----------------------------------------------------------------------------
    vendor = types.MethodType(type(model)._run_denoiser_step_diffusion, model)
    DL.install(model)                                   # no attachments
    ids = tok(build_prompt(spec, spec["problems"][0]["question"]), return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        cs = model._build_context_cache(ids)
        dc = model._build_denoiser_cache_diffusion(cs, dev)
        checks = []
        for frac in (1.0, 0.5):
            blk = torch.full((1, 16), 3, dtype=torch.long, device=dev)
            if frac < 1.0:
                blk[0, :8] = torch.randint(1000, 2000, (8,), device=dev)
            ok, dmax = DL.null_check(model, vendor, blk, cs, torch.tensor([frac], device=dev), dc)
            checks.append(dict(masked_fraction=frac, bitwise_equal=ok, max_abs_diff=dmax))
    res["null_check"] = checks
    print("null check:", checks, flush=True)
    if not all(c["bitwise_equal"] for c in checks):
        json.dump(res, open(a.out + ".json", "w"), indent=1)
        print("NULL CHECK FAILED: the loop copy is not the vendor loop; nothing measured")
        sys.exit(3)

    # ---- shared generation with a stop string ---------------------------------------------------
    orig_extend = model._extend_context_cache
    orig_build = model._build_denoiser_cache_diffusion
    st = {}

    def extend(new_tokens, cache_state, block_wise=True):
        if st.get("time"):
            torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig_extend(new_tokens, cache_state, block_wise=block_wise)
        if st.get("time"):
            torch.cuda.synchronize(); st["extend_s"].append(time.perf_counter() - t0)
        st["ids"].extend(new_tokens[0].tolist())
        st["nfe_per_block"].append(st["nfe_block"]); st["nfe_block"] = 0
        if any(s in tok.decode(st["ids"][-(new_tokens.shape[1] + 8):]) for s in stops):
            raise Stop
        return out

    def build(cache_state, device):
        if st.get("time"):
            torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig_build(cache_state, device)
        if st.get("time"):
            torch.cuda.synchronize(); st["build_s"].append(time.perf_counter() - t0)
        return out

    model._extend_context_cache, model._build_denoiser_cache_diffusion = extend, build

    def generate(question, timing):
        ids = tok(build_prompt(spec, question), return_tensors="pt").input_ids.to(dev)
        st.update(ids=[], nfe_block=0, nfe_per_block=[], time=timing, extend_s=[], build_s=[],
                  step_s=[], call_s=[], last_call=None, prompt_len=int(ids.shape[1]))
        with torch.no_grad():
            try:
                model.generate_mask_diffusion(ids, max_new_tokens=a.max_new_tokens, block_size=16,
                                              steps_per_block=16, mask_token_id=3, temperature=0.0,
                                              confidence_threshold=a.gamma, eos_token_id=tok.eos_token_id)
            except Stop:
                pass

    def wrap_denoise(inner):
        def call(block_ids, cache_state, t=None, den_cache=None):
            st["nfe_block"] += 1
            if st.get("time"):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                if st["last_call"] is not None and st["nfe_block"] > 1:
                    st["step_s"].append(t0 - st["last_call"])      # previous step, sampler included
                st["last_call"] = t0
            out = inner(block_ids, cache_state, t=t, den_cache=den_cache)
            if st.get("time"):
                torch.cuda.synchronize(); st["call_s"].append(time.perf_counter() - t0)
            return out
        return call

    # ---- 1. timing ----------------------------------------------------------------------------------
    timer = {}
    DL.install(model, timer=timer)
    model._run_denoiser_step_diffusion = wrap_denoise(model._run_denoiser_step_diffusion)
    generate(spec["fewshot"][0]["question"], timing=False)               # warm-up (Triton JIT)
    agg = defaultdict(list)
    for p in spec["problems"][:a.n_timing]:
        timer.clear()
        generate(p["question"], timing=True)
        agg["layers_s"] += timer.get("layers_s", [])
        for k in ("step_s", "call_s", "extend_s", "build_s", "nfe_per_block"):
            agg[k] += st[k]
    mean = lambda v: sum(v) / max(len(v), 1)                                  # noqa: E731
    T_step = mean(agg["step_s"])
    tim = dict(passes=len(agg["call_s"]), blocks=len(agg["extend_s"]),
               layers_s=mean(agg["layers_s"]), denoise_call_s=mean(agg["call_s"]), step_s=T_step,
               extend_s=mean(agg["extend_s"]), den_cache_build_s=mean(agg["build_s"]),
               nfe_per_block=mean(agg["nfe_per_block"]))
    tim["f"] = tim["layers_s"] / T_step
    tim["W"] = (tim["extend_s"] + tim["den_cache_build_s"]) / T_step
    tim["note"] = ("step_s = start-to-start of consecutive denoiser calls within a block (the call plus the "
                   "sampler's own work); f = layer loop / step; W = (context-tower extension + denoiser-cache "
                   "build) per block / step. CUDA-synchronised, so absolute times carry the sync cost.")
    res["timing"] = tim
    print("timing:", json.dumps(tim, indent=1), flush=True)

    # ---- 2. calibration + routed experts -------------------------------------------------------------
    cos_sum = [[0.0] * L for _ in range(NB)]
    cos_cnt = [[0] * L for _ in range(NB)]
    cur = {}

    def probe(li, btype, h_in, h_out, den_input):
        m = cur["mask"]
        if not m.any():
            return
        a_ = h_in[0][m].float(); b_ = h_out[0][m].float()
        cos_sum[cur["b"]][li] += float(F.cosine_similarity(a_, b_, dim=-1).mean())
        cos_cnt[cur["b"]][li] += 1

    experts = defaultdict(list)                 # moe layer -> unique experts per pass
    ctx_lens = []

    def gate_hook(li):
        def h(mod, inp, out):
            experts[li].append(int(out[0].unique().numel()))
        return h

    hooks = [layers[i].mixer.gate.register_forward_hook(gate_hook(i))
             for i in range(L) if types_[i] == "moe"]
    DL.install(model, probe=probe)
    inner = model._run_denoiser_step_diffusion

    def calib_call(block_ids, cache_state, t=None, den_cache=None):
        st["nfe_block"] += 1
        m = (block_ids.to(dev) == 3)
        cur["mask"] = m[0]
        cur["b"] = bucket_of(float(m.float().mean()))
        ctx_lens.append(int(cache_state["ctx_len"]))
        return inner(block_ids, cache_state, t=t, den_cache=den_cache)

    model._run_denoiser_step_diffusion = calib_call
    t0 = time.perf_counter()
    for j, p in enumerate(spec["problems"][:a.n_calib]):
        generate(p["question"], timing=False)
        if (j + 1) % 16 == 0:
            print(f"  calibration {j+1}/{a.n_calib} ({time.perf_counter()-t0:.0f}s)", flush=True)
    for h in hooks:
        h.remove()
    mat = [[cos_sum[b][l] / cos_cnt[b][l] if cos_cnt[b][l] else None for l in range(L)] for b in range(NB)]
    glob = [sum(mat[b][l] for b in range(NB) if mat[b][l] is not None)
            / max(sum(mat[b][l] is not None for b in range(NB)), 1) for l in range(L)]
    runs, run = [], []
    for l in range(L):
        if glob[l] > 0.95:
            run.append(l)
        else:
            if run:
                runs.append(run)
            run = []
    if run:
        runs.append(run)
    res["calibration"] = dict(
        buckets=[f"b{b}: r in ({1 - (b + 1) / NB:.2f}, {1 - b / NB:.2f}]" for b in range(NB)],
        cosine=mat, counts=cos_cnt, global_cosine=glob,
        global_order=sorted(range(L), key=lambda l: -glob[l]),
        by_type={t: dict(layers=[l for l in range(L) if types_[l] == t],
                         mean_global_cosine=sum(glob[l] for l in range(L) if types_[l] == t)
                         / sum(1 for l in range(L) if types_[l] == t),
                         by_bucket=[sum(mat[b][l] for l in range(L) if types_[l] == t and mat[b][l] is not None)
                                    / max(sum(1 for l in range(L) if types_[l] == t and mat[b][l] is not None), 1)
                                    for b in range(NB)])
                 for t in sorted(set(types_))},
        runs_above_095=runs, longest_run_above_095=max((len(r) for r in runs), default=0),
        max_cosine=max(glob), argmax_layer=max(range(L), key=lambda l: glob[l]))

    # ---- 3. bytes per layer type --------------------------------------------------------------------
    nbytes = lambda mod: sum(p.numel() * p.element_size() for p in mod.parameters())    # noqa: E731
    with torch.no_grad():
        ids = tok(build_prompt(spec, spec["problems"][0]["question"]), return_tensors="pt").input_ids.to(dev)
        cs = model._build_context_cache(ids)
        dc = orig_build(cs, dev)
    mean_ctx = mean(ctx_lens)
    per_layer = []
    for l, blk in enumerate(layers):
        t = types_[l]
        row = dict(layer=l, type=t, norm_B=nbytes(blk.norm))
        if t == "moe":
            mx = blk.mixer
            one_expert = nbytes(mx.experts[0])
            row.update(router_B=nbytes(mx.gate), shared_B=nbytes(mx.shared_experts), expert_B=one_expert,
                       active_experts_mean=mean(experts[l]),
                       active_experts_max=max(experts[l]) if experts[l] else None,
                       weights_active_B=nbytes(mx.gate) + nbytes(mx.shared_experts) + mean(experts[l]) * one_expert,
                       state_B=0.0)
        elif t == "mamba":
            row.update(weights_active_B=nbytes(blk.mixer),
                       state_B=float(dc.conv_states[l].numel() * dc.conv_states[l].element_size()
                                     + dc.ssm_states[l].numel() * dc.ssm_states[l].element_size()))
        else:
            k = dc.key_cache[l]
            per_tok = 2 * k.numel() * k.element_size() / k.shape[-2]           # K and V, per context token
            row.update(weights_active_B=nbytes(blk.mixer), kv_B_per_ctx_token=per_tok,
                       state_B=per_tok * mean_ctx)
        row["unit_B"] = row["norm_B"] + row["weights_active_B"] + row["state_B"]
        per_layer.append(row)
    total = sum(r["unit_B"] for r in per_layer)
    mean_unit = total / L
    for r in per_layer:
        r["L_eq"] = r["unit_B"] / mean_unit
    MB = 1e6
    sums = lambda key, cond: sum(r[key] for r in per_layer if cond(r))        # noqa: E731
    res["bytes"] = dict(
        mean_ctx_tokens=mean_ctx, per_layer=per_layer, total_MB_per_pass=total / MB, mean_unit_MB=mean_unit / MB,
        by_type={t: dict(n=sum(1 for r in per_layer if r["type"] == t),
                         unit_MB=sums("unit_B", lambda r, t=t: r["type"] == t) / MB
                         / sum(1 for r in per_layer if r["type"] == t),
                         L_eq_per_layer=sum(r["L_eq"] for r in per_layer if r["type"] == t)
                         / sum(1 for r in per_layer if r["type"] == t))
                 for t in sorted(set(types_))},
        stage0_inputs=dict(L=L,
                           attn_MB=sums("weights_active_B", lambda r: r["type"] in ("mamba", "attention")) / L / MB,
                           ffn_MB=sums("weights_active_B", lambda r: r["type"] == "moe") / L / MB,
                           kv_MB=(sums("state_B", lambda r: True) + sums("norm_B", lambda r: True)) / L / MB,
                           note="stage0.py models uniform layers; these are the 52-layer means, so the sum of "
                                "the three is the mean per-pass bytes of one layer. attn_MB carries the Mamba "
                                "and attention mixers, ffn_MB the MoE's active bytes, kv_MB the state reads."))
    json.dump(res, open(a.out + ".json", "w"), indent=1)

    c = res["calibration"]
    print(f"\nstep 3: max global cosine {c['max_cosine']:.4f} at layer {c['argmax_layer']}; "
          f"longest run > 0.95: {c['longest_run_above_095']} layers {c['runs_above_095']}")
    for t, v in c["by_type"].items():
        print(f"  {t:9s} n={len(v['layers']):2d} mean cos {v['mean_global_cosine']:.4f}  by bucket "
              + " ".join(f"{x:.4f}" for x in v["by_bucket"]))
    print("  global order (most redundant first):", c["global_order"][:16])
    b = res["bytes"]
    print(f"\nstep 4 bytes: mean ctx {b['mean_ctx_tokens']:.0f}; total {b['total_MB_per_pass']:.1f} MB/pass; "
          f"mean unit {b['mean_unit_MB']:.2f} MB/layer")
    for t, v in b["by_type"].items():
        print(f"  {t:9s} n={v['n']:2d} unit {v['unit_MB']:.2f} MB  L_eq per layer {v['L_eq_per_layer']:.3f}")
    print("  stage0 inputs:", b["stage0_inputs"])
    print("  f", round(tim["f"], 4), "W", round(tim["W"], 4), "N (measured here)", round(tim["nfe_per_block"], 2))


if __name__ == "__main__":
    main()
