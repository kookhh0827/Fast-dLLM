"""Phase 1b Part C on Nemotron-Labs-TwoTower (`results/phase1b/PREREG.md` C and its 2026-09-14 amendment (e)):
the Family C state x depth map with every Fisher variant on the same canvases. One model load, four stages,
each checkpointed to a state file (ce_hopper preempts):

  bank    the 100 GSM8K-test problems of step 2 generated at full depth (gamma 0.8, S 16, T 16, 8-shot, stop at
          `Q:`), every pass's input canvas recorded; per problem ONE block is drawn (seeded) among the blocks whose
          passes cover all five r-buckets, and its first pass in each bucket is kept, with the block's denoiser
          cache copied to CPU -- the exact cache the generation used, so nothing is rebuilt. A problem with no
          such block takes, per bucket, the earliest block that has one (marked `fallback`).
  map     per canvas: the full pass, the sampler's commit set (the vendor rule at gamma 0.8, step index
          included), then all 52 single-sublayer identity skips: per committed token KL(full || skip) on the
          MDLM distribution (mask token excluded) and argmax agreement; per canvas A2 (i) mean max-probability
          shift on the committed tokens and (ii) Spearman of the max-probability ranking over all masked
          positions. Plus step 5's static MoE sets k3/k5/k7 (epsilon).
  fisher  one forward with grad, then per committed token three backwards with retained graph: y = the
          committed token (F-emp) and two samples y ~ p (F-true). For every sublayer and parameter group, the
          sum of squared parameter gradients (post-accumulate hooks free each gradient at once); for every
          sublayer the activation Fisher on its residual update (grad of the sublayer output == grad of
          delta_l) summed over positions x dims, KL2 = 1/2 sum F-act * delta^2 (diagonal, emp and true), and the
          rank-1 form 1/2 (g . delta)^2 averaged over the true samples (reported as auxiliary).
  greedy  per (r-bucket, half): cumulative KL-greedy over the MoE sublayers (sublayer 51 protected as in step 5,
          no two consecutive in MoE order), to exhaustion; the full model's committed log-probs are cached per
          canvas so every step evaluates only skip passes.

Outputs `<out>.npz` (per token and per canvas arrays) and `<out>.json` (bank summary, traces, groups).
"""
import argparse
import json
import math
import os
import random
import sys
import time
import types as types_mod
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step2_repro import Stop, build_prompt, extract, normalise    # noqa: E402
import denoiser_loop as DL                                         # noqa: E402

MASK = 3
S_BLK = 16
BUCKETS = [("(0.8, 1.0]", 0.8, 1.01), ("(0.6, 0.8]", 0.6, 0.8), ("(0.4, 0.6]", 0.4, 0.6),
           ("(0.2, 0.4]", 0.2, 0.4), ("[0.0, 0.2]", -0.01, 0.2)]
GROUPS = ["in_proj", "out_proj", "conv1d", "ssm", "mixer_norm", "q", "k", "v", "o",
          "router", "experts", "shared", "norm", "adaln"]
STATIC = {"k3": [1, 8, 15], "k5": [1, 8, 15, 20, 45], "k7": None}      # k7 read from step5_cells.json


def bucket_of(r):
    for i, (_, lo, hi) in enumerate(BUCKETS):
        if lo < r <= hi:
            return i
    raise ValueError(r)


def group_of(name):
    """Parameter name inside `denoiser_tower.layers.<l>.` -> group."""
    if name.startswith("norm."):
        return "norm"
    n = name[len("mixer."):]
    for pre, g in (("in_proj", "in_proj"), ("out_proj", "out_proj"), ("conv1d", "conv1d"), ("norm", "mixer_norm"),
                   ("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "o"), ("gate", "router"),
                   ("experts", "experts"), ("shared_experts", "shared")):
        if n.startswith(pre):
            return g
    if n in ("A_log", "D", "dt_bias"):
        return "ssm"
    raise KeyError(name)


def cache_to_cpu(den):
    return {k: [t.detach().to("cpu", copy=True) if torch.is_tensor(t) else t for t in getattr(den, k)]
            for k in ("conv_states", "ssm_states", "key_cache", "value_cache")}


def cache_to_gpu(model, blob, dev):
    den = model._make_cache(model.config, 1, model.dtype, dev)
    for k, v in blob.items():
        setattr(den, k, [t.to(dev) if torch.is_tensor(t) else t for t in v])
    den.has_previous_state = True
    return den


def mdlm_lp(logits):
    """log p(x0 | xt) over the vocabulary with the mask token excluded (the vendor's `_mdlm_forward` at a masked
    position), batch row 0, float32."""
    lg = logits[0].float()
    return torch.log_softmax(lg.masked_fill(torch.arange(lg.shape[-1], device=lg.device) == MASK, -1e12), -1)


def commit_set(logits, xt, step_idx, gamma):
    """The vendor rule of `generate_mask_diffusion` (B = 1): returns (log p over vocab with the mask token
    excluded, masked bool, committed bool, max-prob, argmax)."""
    lp = mdlm_lp(logits)
    masked = (xt[0] == MASK)
    mp, arg = lp.exp().max(-1)
    n_m = int(masked.sum())
    if step_idx == S_BLK - 1:
        tc = n_m
    else:
        above = int(((mp > gamma) & masked).sum())
        tc = max(above if above > 0 else 1, math.ceil(n_m / max(1, S_BLK - step_idx)))
        tc = min(tc, n_m)
    conf = torch.where(masked, mp, torch.full_like(mp, -1.0))
    com = torch.zeros_like(masked)
    com[conf.topk(tc).indices] = True
    return lp, masked, com, mp, arg


def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra -= ra.mean(); rb -= rb.mean()
    d = ra.norm() * rb.norm()
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--step5-cells", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=None, help="smoke: problems")
    ap.add_argument("--kmax", type=int, default=11)
    ap.add_argument("--n-true", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()

    sys.path.insert(0, a.model)
    from modeling_nemotron_twotower import NemotronHTwoTowerForCausalLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = NemotronHTwoTowerForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                         trust_remote_code=True).cuda().eval()
    model.requires_grad_(False)
    dev = next(model.context_tower.parameters()).device
    layers = model.denoiser_tower.layers
    L = len(layers)
    types = [b.block_type for b in layers]
    spec = json.load(open(a.problems))
    problems = spec["problems"][:a.limit] if a.limit else spec["problems"]
    stops = spec["until"]
    cells5 = {c["name"]: c["skip"] for c in json.load(open(a.step5_cells))["cells"]}
    STATIC["k7"] = cells5["k7_g0.8"]
    assert cells5["k3_g0.8"] == STATIC["k3"] and cells5["k5_g0.8"] == STATIC["k5"]
    st = torch.load(a.state, weights_only=False) if os.path.exists(a.state) else {}

    def save():
        tmp = a.state + ".tmp"
        torch.save(st, tmp)
        os.replace(tmp, a.state)

    # ---- bank --------------------------------------------------------------------------------------------
    if "bank" not in st:
        DL.install(model)
        g = {}
        orig_extend = model._extend_context_cache
        orig_build = model._build_denoiser_cache_diffusion

        def build(cache_state, device):
            g["den"] = orig_build(cache_state, device)
            g["ctx_len"] = int(cache_state["ctx_len"])
            return g["den"]

        def extend(new_tokens, cache_state, block_wise=True):
            passes = g["passes"]; g["passes"] = []
            bk = {}
            for p in passes:
                bk.setdefault(p["bucket"], p)
            g["blocks"].append(dict(block=len(g["blocks"]), buckets=sorted(bk), first=bk, n_passes=len(passes),
                                    cache=cache_to_cpu(g["den"]) if (len(bk) == 5 or any(b not in g["seen"] for b in bk)) else None,
                                    ctx_len=g["ctx_len"]))
            g["seen"] |= set(bk)
            out = orig_extend(new_tokens, cache_state, block_wise=block_wise)
            g["ids"].extend(new_tokens[0].tolist())
            if any(s in tok.decode(g["ids"][-(new_tokens.shape[1] + 8):]) for s in stops):
                raise Stop
            return out

        def cb(step_idx, steps, xt, t=None, logits=None, block_idx=None):
            if logits is None:
                return
            n_m = int((xt == MASK).sum())
            g["passes"].append(dict(xt=xt[0].to("cpu", torch.int32).clone(), t=float(t), step=int(step_idx),
                                    r=n_m / S_BLK, bucket=bucket_of(n_m / S_BLK)))

        model._extend_context_cache, model._build_denoiser_cache_diffusion = extend, build
        bank, summary, caches = [], [], {}
        t0 = time.perf_counter()
        for pi, p in enumerate(problems):
            g.update(passes=[], blocks=[], ids=[], seen=set())
            ids = tok(build_prompt(spec, p["question"]), return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                try:
                    model.generate_mask_diffusion(ids, max_new_tokens=a.max_new_tokens, block_size=S_BLK,
                                                  steps_per_block=S_BLK, mask_token_id=MASK, temperature=0.0,
                                                  confidence_threshold=a.gamma, eos_token_id=tok.eos_token_id,
                                                  step_callback=cb)
                except Stop:
                    pass
            text = tok.decode(g["ids"], skip_special_tokens=True)
            cut = min([text.find(s) for s in stops if s in text] or [len(text)])
            _, flex = extract(text[:cut])
            correct = normalise(flex) == normalise(p["gold"])
            full5 = [b for b in g["blocks"] if len(b["buckets"]) == 5]
            if full5:
                blk = random.Random(1000 + pi).choice(full5)
                chosen = [(bi, blk) for bi in range(5)]
                mode = "one_block"
            else:
                chosen = []
                for bi in range(5):
                    cand = [b for b in g["blocks"] if bi in b["first"]]
                    if cand:
                        chosen.append((bi, cand[0]))
                mode = "fallback"
            keys = {}
            for bi, blk in chosen:
                if blk["block"] not in keys:
                    keys[blk["block"]] = f"{pi}:{blk['block']}"
                    caches[keys[blk["block"]]] = blk["cache"]
                pas = blk["first"][bi]
                bank.append(dict(problem=pi, pid=p["id"], half=pi % 2, bucket=bi, block=blk["block"],
                                 step=pas["step"], r=pas["r"], t=pas["t"], xt=pas["xt"], cache=keys[blk["block"]],
                                 ctx_len=blk["ctx_len"], mode=mode))
            summary.append(dict(problem=pi, pid=p["id"], correct=bool(correct), n_blocks=len(g["blocks"]),
                                n_full5_blocks=len(full5), mode=mode, blocks_used=sorted(keys)))
            if (pi + 1) % 10 == 0:
                print(f"  bank: {pi+1}/{len(problems)} problems, {len(bank)} canvases "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
        model._extend_context_cache, model._build_denoiser_cache_diffusion = orig_extend, orig_build
        torch.save(caches, a.state + ".caches.pt")
        st["bank"], st["bank_summary"] = bank, summary
        save()
        print(f"bank: {len(bank)} canvases, {sum(s['mode']=='fallback' for s in summary)} fallback problems, "
              f"accuracy {np.mean([s['correct'] for s in summary]):.3f}", flush=True)
    bank = st["bank"]
    N = len(bank)
    caches = torch.load(a.state + ".caches.pt", weights_only=False)
    gpu_cache = {}

    def den_for(c):
        if c["cache"] not in gpu_cache:
            gpu_cache.clear()
            gpu_cache[c["cache"]] = cache_to_gpu(model, caches[c["cache"]], dev)
        return gpu_cache[c["cache"]]

    # the loop copy must still be the vendor loop on a bank canvas (step 3's check, repeated here)
    vendor = types_mod.MethodType(type(model)._run_denoiser_step_diffusion, model)
    DL.install(model)
    c0 = bank[0]
    ok, dmax = DL.null_check(model, vendor, c0["xt"].to(dev, torch.long)[None], {"ctx_len": c0["ctx_len"]},
                             torch.tensor([c0["t"]], device=dev), den_for(c0))
    print(f"null check on canvas 0: bitwise {ok} (max |diff| {dmax})", flush=True)
    if not ok:
        sys.exit(3)

    probe_state = {}

    def probe(li, btype, h_in, h_out, den_input):
        if probe_state.get("on"):
            h_out.retain_grad()
            probe_state["h"][li] = h_out
            probe_state["d"][li] = (h_out.detach().float() - h_in.detach().float())

    def run(c, skip=None, grad=False):
        DL.install(model, probe=probe if grad else None, skip=set(skip) if skip else None)
        xt = c["xt"].to(dev, torch.long)[None]
        t = torch.tensor([c["t"]], device=dev)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            return model._run_denoiser_step_diffusion(xt, {"ctx_len": c["ctx_len"]}, t=t, den_cache=den_for(c)), xt

    # ---- map ---------------------------------------------------------------------------------------------
    if not st.get("map_done"):
        m = st.setdefault("map", dict(done=0, tok=[], canv=[]))
        t0 = time.perf_counter()
        for ci in range(m["done"], N):
            c = bank[ci]
            lg, xt = run(c)
            lp, masked, com, mp, arg = commit_set(lg, xt, c["step"], a.gamma)
            pos = com.nonzero().squeeze(-1)
            pf = lp[pos].exp()
            kl = np.zeros((len(pos), L + 3), np.float32); ag = np.zeros((len(pos), L + 3), np.int8)
            sp = np.full(L + 3, np.nan, np.float32); ds = np.zeros(L + 3, np.float32)
            configs = [[l] for l in range(L)] + [STATIC["k3"], STATIC["k5"], STATIC["k7"]]
            for j, sk in enumerate(configs):
                lgs, _ = run(c, skip=sk)
                lps = mdlm_lp(lgs)
                kl[:, j] = (pf * (lp[pos] - lps[pos])).sum(-1).cpu().numpy()
                ag[:, j] = (lps[pos].argmax(-1) == arg[pos]).cpu().numpy()
                mps = lps.exp().max(-1).values
                if int(masked.sum()) >= 3:
                    sp[j] = spearman(mp[masked], mps[masked])
                ds[j] = float((mps[pos] - mp[pos]).mean())
            m["tok"].append(dict(canvas=ci, pos=pos.cpu().numpy(), conf=mp[pos].cpu().numpy(),
                                 tokid=arg[pos].cpu().numpy(), kl=kl, agree=ag))
            m["canv"].append(dict(canvas=ci, n_mask=int(masked.sum()), n_commit=len(pos), spearman=sp, dshift=ds))
            m["done"] = ci + 1
            if (ci + 1) % 50 == 0 or ci + 1 == N:
                print(f"  map: {ci+1}/{N} ({time.perf_counter()-t0:.0f}s this allocation)", flush=True)
                save()
        st["map_done"] = True
        save()

    # ---- fisher ------------------------------------------------------------------------------------------
    if not st.get("fisher_done"):
        den_params = []                            # (layer, group, param)
        for l, blk in enumerate(layers):
            for name, p in blk.named_parameters():
                den_params.append((l, group_of(name), p))
        for l in range(L):
            den_params.append((l, "adaln", model.scale_shift_tables[l]))
        gi = {g: i for i, g in enumerate(GROUPS)}
        acc = torch.zeros(L, len(GROUPS), device=dev, dtype=torch.float64)
        numel = np.zeros((L, len(GROUPS)), np.int64); elsize = np.zeros((L, len(GROUPS)), np.int64)
        expert_numel = np.zeros(L, np.int64)
        for l, grp, p in den_params:
            p.requires_grad_(True)
            numel[l, gi[grp]] += p.numel(); elsize[l, gi[grp]] = p.element_size()

            def hook(param, l=l, j=gi[grp]):
                acc[l, j] += param.grad.detach().double().pow(2).sum()
                param.grad = None
            p.register_post_accumulate_grad_hook(hook)
        for l, blk in enumerate(layers):
            if types[l] == "moe":
                expert_numel[l] = sum(q.numel() for q in blk.mixer.experts[0].parameters())
        routed = {}

        def gate_hook(l):
            def h(mod, inp, out):
                routed[l] = int(out[0].unique().numel())
            return h
        hooks = [layers[l].mixer.gate.register_forward_hook(gate_hook(l)) for l in range(L) if types[l] == "moe"]
        fz = st.setdefault("fisher", dict(done=0, rows=[]))
        gen = torch.Generator(device=dev)
        t0 = time.perf_counter()
        for ci in range(fz["done"], N):
            c = bank[ci]
            gen.manual_seed(ci)
            probe_state.update(on=True, h={}, d={})
            routed.clear()
            lg, xt = run(c, grad=True)
            probe_state["on"] = False
            with torch.no_grad():
                _, masked, com, mp, arg = commit_set(lg.detach(), xt, c["step"], a.gamma)
            pos = com.nonzero().squeeze(-1).tolist()
            lp = mdlm_lp(lg)
            T = len(pos)
            Femp = np.zeros((T, L, len(GROUPS)), np.float64); Ftrue = np.zeros_like(Femp)
            Aemp = np.zeros((T, L)); Atrue = np.zeros((T, L)); K2emp = np.zeros((T, L)); K2true = np.zeros((T, L))
            Q1true = np.zeros((T, L))
            d2 = {l: probe_state["d"][l].pow(2) for l in range(L)}
            for ti, j in enumerate(pos):
                probs = lp[j].detach().exp()
                targets = [("emp", int(arg[j]))] + [("true", int(torch.multinomial(probs, 1, generator=gen)))
                                                    for _ in range(a.n_true)]
                for kind, y in targets:
                    for l in range(L):
                        probe_state["h"][l].grad = None
                    acc.zero_()
                    lp[j, y].backward(retain_graph=True)
                    with torch.no_grad():
                        gsq = torch.stack([probe_state["h"][l].grad.float().pow(2).sum() for l in range(L)])
                        k2 = torch.stack([0.5 * (probe_state["h"][l].grad.float().pow(2) * d2[l]).sum()
                                          for l in range(L)])
                        q1 = torch.stack([0.5 * (probe_state["h"][l].grad.float() * probe_state["d"][l]).sum().pow(2)
                                          for l in range(L)])
                        vals = (acc.cpu().numpy(), gsq.double().cpu().numpy(), k2.double().cpu().numpy(),
                                q1.double().cpu().numpy())
                    if kind == "emp":
                        Femp[ti] = vals[0]; Aemp[ti] = vals[1]; K2emp[ti] = vals[2]
                    else:
                        Ftrue[ti] += vals[0] / a.n_true; Atrue[ti] += vals[1] / a.n_true
                        K2true[ti] += vals[2] / a.n_true; Q1true[ti] += vals[3] / a.n_true
            probe_state.update(h={}, d={})
            del lg, lp
            fz["rows"].append(dict(canvas=ci, pos=np.array(pos), F_emp=Femp.astype(np.float32),
                                   F_true=Ftrue.astype(np.float32), Fact_emp=Aemp.astype(np.float32),
                                   Fact_true=Atrue.astype(np.float32), KL2_emp=K2emp.astype(np.float32),
                                   KL2_true=K2true.astype(np.float32), KL2q_true=Q1true.astype(np.float32),
                                   routed=np.array([routed.get(l, 0) for l in range(L)], np.int16)))
            fz["done"] = ci + 1
            if (ci + 1) % 25 == 0 or ci + 1 == N:
                print(f"  fisher: {ci+1}/{N} ({time.perf_counter()-t0:.0f}s)", flush=True)
                save()
        for h in hooks:
            h.remove()
        st["fisher_meta"] = dict(groups=GROUPS, numel=numel, element_size=elsize, expert_numel=expert_numel)
        st["fisher_done"] = True
        save()
        torch.cuda.empty_cache()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- greedy ------------------------------------------------------------------------------------------
    moe = [l for l in range(L) if types[l] == "moe"]
    moe_pos = {l: j for j, l in enumerate(moe)}
    pool0 = [l for l in moe if l != L - 1]
    greedy = st.setdefault("greedy", {})
    for bi in range(5):
        for h in (0, 1):
            key = f"b{bi}h{h}"
            if key in greedy:
                continue
            cs = [ci for ci, c in enumerate(bank) if c["bucket"] == bi and c["half"] == h]
            refs = {}
            for ci in cs:
                lg, xt = run(bank[ci])
                lp, masked, com, mp, arg = commit_set(lg, xt, bank[ci]["step"], a.gamma)
                refs[ci] = (com, lp[com])
            chosen, trace = [], []
            t0 = time.perf_counter()
            for _ in range(a.kmax):
                pool = [l for l in pool0 if l not in chosen and not any(abs(moe_pos[l] - moe_pos[x]) == 1 for x in chosen)]
                if not pool:
                    break
                tot = {l: 0.0 for l in pool}
                per = {l: [] for l in pool}
                for ci in cs:
                    com, lpc = refs[ci]
                    for l in pool:
                        lgs, _ = run(bank[ci], skip=chosen + [l])
                        lps = mdlm_lp(lgs)[com]
                        v = float((lpc.exp() * (lpc - lps)).sum(-1).mean())
                        tot[l] += v; per[l].append(v)
                best = min(pool, key=lambda l: tot[l])
                chosen.append(best)
                trace.append(dict(k=len(chosen), chosen=best, set=sorted(chosen), kl=tot[best] / len(cs),
                                  per_canvas=per[best], canvases=cs,
                                  runner_up=sorted(pool, key=lambda l: tot[l])[1] if len(pool) > 1 else None))
                print(f"  greedy {key}: k {len(chosen)} +{best} kl {tot[best]/len(cs):.4g} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
            greedy[key] = dict(n_canvases=len(cs), trace=trace)
            save()

    # ---- write -------------------------------------------------------------------------------------------
    mt, mc, fr = st["map"]["tok"], st["map"]["canv"], st["fisher"]["rows"]
    assert [r["canvas"] for r in fr] == [r["canvas"] for r in mt]
    for r, q in zip(fr, mt):
        assert np.array_equal(r["pos"], q["pos"]), (r["canvas"], r["pos"], q["pos"])
    cat = lambda rows, k: np.concatenate([r[k] for r in rows])                          # noqa: E731
    meta = st["fisher_meta"]
    np.savez_compressed(
        a.out + ".npz",
        tok_canvas=np.concatenate([np.full(len(r["pos"]), r["canvas"], np.int32) for r in mt]),
        tok_pos=cat(mt, "pos"), tok_conf=cat(mt, "conf"), tok_id=cat(mt, "tokid"),
        kl=cat(mt, "kl"), agree=cat(mt, "agree"),
        F_emp=cat(fr, "F_emp"), F_true=cat(fr, "F_true"), Fact_emp=cat(fr, "Fact_emp"), Fact_true=cat(fr, "Fact_true"),
        KL2_emp=cat(fr, "KL2_emp"), KL2_true=cat(fr, "KL2_true"), KL2q_true=cat(fr, "KL2q_true"),
        c_routed=np.stack([r["routed"] for r in fr]),
        c_spearman=np.stack([r["spearman"] for r in mc]), c_dshift=np.stack([r["dshift"] for r in mc]),
        c_n_mask=np.array([r["n_mask"] for r in mc]), c_n_commit=np.array([r["n_commit"] for r in mc]),
        c_problem=np.array([c["problem"] for c in bank]), c_half=np.array([c["half"] for c in bank]),
        c_bucket=np.array([c["bucket"] for c in bank]), c_block=np.array([c["block"] for c in bank]),
        c_step=np.array([c["step"] for c in bank]), c_r=np.array([c["r"] for c in bank]),
        numel=meta["numel"], element_size=meta["element_size"], expert_numel=meta["expert_numel"])
    json.dump(dict(model=a.model, gpu=torch.cuda.get_device_name(0), args=vars(a), layer_types=types,
                   configs=[f"L{l}" for l in range(L)] + ["k3", "k5", "k7"], static_sets=STATIC, groups=GROUPS,
                   buckets=[b[0] for b in BUCKETS], bank_summary=st["bank_summary"], greedy=greedy,
                   greedy_pool=pool0, n_canvases=N),
              open(a.out + ".json", "w"), indent=1)
    print(f"written {a.out}.npz / .json")


if __name__ == "__main__":
    main()
