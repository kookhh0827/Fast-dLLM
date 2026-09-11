"""Phase 1, second pass — everything `results/phase1/PREREG.md` §2 asks for that `phase1_map.py`
did not produce, on the SAME bank (read from its state file, never rebuilt: Family A's
nondeterminism gave 3293 canvases on one build and 3315 on the next, so a rebuilt bank is a
different sample).

What the first pass left out, and why this exists:

  * **epsilon.** §2 reads k* at "the pooled KL of the static identity-kl:7 set"; the map run
    never evaluated that set.
  * **Split halves for the greedy.** §3 needs k* "in both split halves, with the same ordering";
    the first greedy ran once per bucket. It also took the first 100 canvases of each bucket in
    bank order, and the bank is ordered by problem, so each bucket's set was searched on ~20
    problems. Here each (bucket, half) greedy draws its canvases at random from all 64 problems
    of that half.
  * **Terciles and token classes.** The map was split by bucket and half only. Everything here
    is stored per committed token, so any split is a reduction in the reader.
  * **Layers 24-30.** The map excluded the greedy's protected last 8 layers; §2 says "first and
    last excluded", a 30 x 5 map.
  * **A2 and A3** (agreement vs k by tercile and class; tau_eff, commit-set Jaccard, logit lens).

The greedy keeps §2's pool (cosine top-16, standard rules, k up to 12). That pool cannot reach
k = 12: sixteen consecutive layers hold at most eight non-adjacent ones, and the first run
exhausted at 6-8. When the registered pool is exhausted the search continues on every
unprotected layer (1-23) under the same rules, and every such step is marked `pool: extended`
so the reader can say which k* came from where.

Output: one compressed .npz of per-token and per-canvas arrays, plus the greedy traces as JSON.

    python phase1_tok.py --bank-state /scratch1/.../map_A.state.pt --out results/phase1/tok_A
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import phase1_map as M                                                  # noqa: E402
from model.modeling_llada import LLaDAModelLM                           # noqa: E402
from transformers import AutoTokenizer                                  # noqa: E402
from dllm_skip.hook import install_skipping, uninstall_skipping         # noqa: E402

CLASSES = ["digit", "operator", "other"]


@torch.no_grad()
def reference_lens(model, c, thr):
    """`M.reference` plus the hidden states of the block pass, for the logit lens."""
    pkv = model(c["x"], use_cache=True).past_key_values
    out = model(c["x"][:, c["s"]:c["e"]], past_key_values=pkv, use_cache=True,
                replace_position=c["rp"], output_hidden_states=True)
    lg = out.logits
    mb = (c["x"][:, c["s"]:c["e"]] == M.MASK_ID)
    _x0, ti, conf = M.G.get_transfer_index(lg, 0.0, "low_confidence", mb, c["x"][:, c["s"]:c["e"]],
                                           None, thr, return_confidence=True)
    return pkv, lg, ti, conf, mb, out.hidden_states


@torch.no_grad()
def crystallisation(model, hs, final_arg, pos):
    """Per position in `pos`: the first layer (0-indexed, "after block l") from which the logit
    lens argmax equals the final argmax at every later layer. hidden_states holds the input to
    each block and, last, ln_f of the final block's output (HF convention), so the output of
    block l < 31 is hs[l + 1] and needs ln_f; block 31's is hs[32] as is."""
    tr = model.model.transformer
    cfg = model.model.config
    W = tr.wte.weight if cfg.weight_tying else tr.ff_out.weight
    n = len(hs) - 1                                   # number of blocks
    agree = []
    for l in range(n):
        h = hs[l + 1][0, pos]
        if l < n - 1:
            h = tr.ln_f(h)
        agree.append(torch.nn.functional.linear(h, W).argmax(-1) == final_arg)
    agree = torch.stack(agree)                        # (n, P)
    # last layer from the top at which it DISagrees; crystallised one above it
    out = torch.zeros(agree.shape[1], dtype=torch.long, device=agree.device)
    for l in range(n - 1, -1, -1):
        bad = ~agree[l] & (out == 0)
        out[bad] = l + 1
    return out                                        # 0 means "agrees from layer 0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--calib", default="/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json")
    ap.add_argument("--kl-json", default="/home1/hyunhoko/DLLM/results/phase0.25/klgreedy_A.json",
                    help="the static identity-kl:7 set and its greedy order (epsilon, A2, A3)")
    ap.add_argument("--bank-state", required=True, help="phase1_map.py's state file (the bank)")
    ap.add_argument("--out", required=True, help="prefix: writes <out>.npz and <out>.json")
    ap.add_argument("--state", default=None, help="this pass's checkpoint; default <out>.state.pt")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--keep-first", type=int, default=1)
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--n-cands", type=int, default=16)
    ap.add_argument("--greedy-canvases", type=int, default=100, help="per (bucket, half)")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: use only the first N canvases of the bank")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    cal = json.load(open(a.calib))
    L = cal["n_layers"]
    kl = json.load(open(a.kl_json))
    static_order = [t["chosen"] for t in kl["trace"]]
    assert sorted(static_order) == sorted(kl["kl_greedy_set"]), (static_order, kl["kl_greedy_set"])

    bank = torch.load(a.bank_state, weights_only=False)["bank"]
    if a.limit:
        bank = bank[:a.limit]
    print(f"bank: {len(bank)} canvases from {a.bank_state} (not rebuilt)", flush=True)
    half = {c["problem"]: c["problem"] % 2 for c in bank}

    protected = set(range(a.keep_first)) | set(range(L - max(a.keep_last, 1), L))
    cands = [l for l in cal["global_order"] if l not in protected][:a.n_cands]
    ext_pool = [l for l in range(L) if l not in protected]
    ctrl = install_skipping(model)

    state_path = a.state or (a.out + ".state.pt")
    st = torch.load(state_path, weights_only=False) if os.path.exists(state_path) else {}

    # ---- A: the split-half per-bucket greedy ------------------------------------------------
    def greedy_for(canvases):
        chosen, trace = [], []
        while len(chosen) < a.kmax:
            def admissible(pool):
                return [l for l in pool if l not in chosen and not any(abs(l - c) == 1 for c in chosen)]
            pool, tag = admissible(cands), "registered"
            if not pool:
                pool, tag = admissible(ext_pool), "extended"
            if not pool:
                trace.append(dict(step=len(chosen) + 1, exhausted=True, set=sorted(chosen)))
                break
            tot = {l: 0.0 for l in pool}
            n = 0
            for c in canvases:
                pkv, ref_lg, ti, _conf, _arg = M.reference(model, c, a.threshold)
                if not ti.any():
                    continue
                n += 1
                for l in pool:
                    keep = [j for j in range(L) if j not in chosen + [l]]
                    tot[l] += M.kl_on(ref_lg, M.skip_pass(model, ctrl, c, keep, pkv), ti)
            best = min(pool, key=lambda l: tot[l])
            chosen.append(best)
            trace.append(dict(step=len(chosen), chosen=best, kl=tot[best] / max(n, 1), pool=tag,
                              set=sorted(chosen), n_canvases=n,
                              all_scores={str(l): tot[l] / max(n, 1) for l in pool}))
        return trace

    greedy = dict(st.get("greedy", {}))
    for bi, (bname, _lo, _hi) in enumerate(M.BUCKETS):
        for h in (0, 1):
            key = f"{bi}|{h}"
            if key in greedy:
                print(f"  greedy {bname} half{h}: already done", flush=True)
                continue
            idx = [i for i, c in enumerate(bank) if c["bucket"] == bi and half[c["problem"]] == h]
            if not idx:
                continue
            rng = random.Random(1000 * bi + h)
            pick = sorted(rng.sample(idx, min(a.greedy_canvases, len(idx))))
            t0 = time.perf_counter()
            tr = greedy_for([bank[i] for i in pick])
            greedy[key] = dict(bucket=bname, half=h, canvas_idx=pick,
                               n_problems=len({bank[i]["problem"] for i in pick}), trace=tr)
            st["greedy"] = greedy
            torch.save(st, state_path)
            print(f"  greedy {bname} half{h}: {[t.get('chosen') for t in tr]} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---- B: per-token records on every canvas ----------------------------------------------
    configs = [f"L{l}" for l in range(L) if l != 0 and l != L - 1]                 # 30-layer map
    configs += [f"S{k}" for k in range(1, len(static_order) + 1)]                   # static prefixes
    seq = {}                                                                         # (b, h) -> order
    for key, g in greedy.items():
        bi, h = map(int, key.split("|"))
        seq[(bi, h)] = [t["chosen"] for t in g["trace"] if "chosen" in t]
        configs += [f"G{bi}h{h}k{k}" for k in range(1, len(seq[(bi, h)]) + 1)]
    cid = {n: i for i, n in enumerate(configs)}

    def skipset(name, bucket):
        if name[0] == "L":
            return [int(name[1:])]
        if name[0] == "S":
            return static_order[:int(name[1:])]
        bi, rest = name[1:].split("h")
        h, k = rest.split("k")
        return seq[(int(bi), int(h))][:int(k)] if int(bi) == bucket else None

    recs = st.get("recs", [])
    start = len(recs)
    if start:
        print(f"resumed records at canvas {start}/{len(bank)}", flush=True)
    cls_cache = {}
    t0 = time.perf_counter()
    for ci in range(start, len(bank)):
        c = bank[ci]
        pkv, ref_lg, ti, conf, mb, hs = reference_lens(model, c, a.threshold)
        m = ti[0]
        mask = mb[0]
        ref_arg = ref_lg[0].argmax(-1)
        pos = m.nonzero().squeeze(-1)
        n_full = int(m.sum())
        assert n_full >= 1, ci                       # the forced-argmax rule commits >= 1
        cry = crystallisation(model, hs, ref_arg[pos], pos)
        ids = ref_arg[pos].tolist()
        for t in ids:
            if t not in cls_cache:
                cls_cache[t] = CLASSES.index(M.token_class(tok, t))
        rec = dict(ci=ci, bucket=c["bucket"], half=half[c["problem"]], problem=c["problem"],
                   block=c["block"], step=c["step"], r=c["r"], n_mask=int(mask.sum()),
                   n_commit=n_full, conf=conf[0][pos].float().cpu(),
                   cls=torch.tensor([cls_cache[t] for t in ids], dtype=torch.int8),
                   cryst=cry.to(torch.int8).cpu(), cfg={},
                   # k = 0 row of A3's max-prob distribution: `conf` holds the max-probability on
                   # every masked position (temperature 0, argmax proposal)
                   mp_full=conf[0][mask].float().cpu())
        p = torch.log_softmax(ref_lg[0][pos].float(), -1)
        full_set = set(pos.tolist())
        for name in configs:
            sk = skipset(name, c["bucket"])
            if sk is None:
                continue
            keep = [j for j in range(L) if j not in sk]
            lg = M.skip_pass(model, ctrl, c, keep, pkv)[0]
            q = torch.log_softmax(lg[pos].float(), -1)
            klv = (p.exp() * (p - q)).sum(-1)
            ag = lg[pos].argmax(-1) == ref_arg[pos]
            # A3: the skip pass's own max-prob on every masked position, its commit count at the
            # gate threshold, tau_eff = the threshold that reproduces the full model's count,
            # and the Jaccard between that top-n set and the full model's commit set.
            mp = torch.softmax(lg[mask].double(), -1).max(-1).values
            mpos = mask.nonzero().squeeze(-1)
            order = torch.argsort(mp, descending=True)
            top = set(mpos[order[:n_full]].tolist())
            n09 = max(int((mp >= a.threshold).sum()), 1)
            rec["cfg"][cid[name]] = dict(
                kl=klv.cpu(), agree=ag.cpu(), n09=n09,
                tau_eff=float(mp[order[n_full - 1]]),
                jac=len(top & full_set) / len(top | full_set),
                maxprob=mp.float().cpu() if name[0] == "S" else None)
        recs.append(rec)
        if (ci + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  records: {ci+1}/{len(bank)} ({el/(ci+1-start):.2f}s/canvas)", flush=True)
        if (ci + 1) % a.save_every == 0:
            st["recs"] = recs
            torch.save(st, state_path)
    st["recs"] = recs
    torch.save(st, state_path)
    uninstall_skipping(model)

    # ---- flatten ---------------------------------------------------------------------------
    C, T = len(configs), sum(len(r["conf"]) for r in recs)
    tok_canvas = np.concatenate([np.full(len(r["conf"]), j, np.int32) for j, r in enumerate(recs)])
    kl_a = np.full((C, T), np.nan, np.float32)
    ag_a = np.full((C, T), -1, np.int8)
    n09 = np.full((C, len(recs)), -1, np.int32)
    taue = np.full((C, len(recs)), np.nan, np.float32)
    jac = np.full((C, len(recs)), np.nan, np.float32)
    mp_vals, mp_canvas, mp_cfg = [], [], []
    off = 0
    for j, r in enumerate(recs):
        nt = len(r["conf"])
        mp_vals.append(r["mp_full"].numpy().astype(np.float16))                # cfg -1 = full depth
        mp_canvas.append(np.full(len(r["mp_full"]), j, np.int32))
        mp_cfg.append(np.full(len(r["mp_full"]), -1, np.int16))
        for ic, d in r["cfg"].items():
            kl_a[ic, off:off + nt] = d["kl"].numpy()
            ag_a[ic, off:off + nt] = d["agree"].numpy().astype(np.int8)
            n09[ic, j], taue[ic, j], jac[ic, j] = d["n09"], d["tau_eff"], d["jac"]
            if d["maxprob"] is not None:
                mp_vals.append(d["maxprob"].numpy().astype(np.float16))
                mp_canvas.append(np.full(len(d["maxprob"]), j, np.int32))
                mp_cfg.append(np.full(len(d["maxprob"]), ic, np.int16))
        off += nt
    cv = lambda k, dt: np.array([r[k] for r in recs], dt)                       # noqa: E731
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out + ".npz", configs=np.array(configs), kl=kl_a, agree=ag_a, n09=n09, tau_eff=taue,
        jac=jac, tok_canvas=tok_canvas,
        tok_conf=np.concatenate([r["conf"].numpy() for r in recs]).astype(np.float32),
        tok_cls=np.concatenate([r["cls"].numpy() for r in recs]),
        tok_cryst=np.concatenate([r["cryst"].numpy() for r in recs]),
        c_bank_idx=cv("ci", np.int32), c_bucket=cv("bucket", np.int8), c_half=cv("half", np.int8),
        c_problem=cv("problem", np.int16), c_block=cv("block", np.int8),
        c_step=cv("step", np.int16), c_r=cv("r", np.float32), c_n_mask=cv("n_mask", np.int16),
        c_n_commit=cv("n_commit", np.int16),
        mp_vals=np.concatenate(mp_vals), mp_canvas=np.concatenate(mp_canvas),
        mp_cfg=np.concatenate(mp_cfg))
    json.dump(dict(model=a.model, L=L, n_canvases=len(recs), n_tokens=T, args=vars(a),
                   buckets=[b[0] for b in M.BUCKETS], classes=CLASSES, candidates=cands,
                   extended_pool=ext_pool, static_order=static_order, configs=configs,
                   greedy=greedy), open(a.out + ".json", "w"), indent=1)
    print(f"written {a.out}.npz / .json  ({len(recs)} canvases, {T} committed tokens, "
          f"{C} configs)")


if __name__ == "__main__":
    main()
