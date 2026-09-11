"""Phase 1, Family B (Fast-dLLM v2 7B) — `results/phase1/PREREG.md` §1-2, the same measurements
as `v1/llada/phase1_tok.py` and the same output layout, so one reader serves both families.

**Snapshots.** §1: "Family B: rebuild [the cache] by prefill + encodes". Family B's exact cache is
written only by the prompt prefill and each clean-block encode; a refinement pass reads it and
never writes (`update_past_key_values=False` concatenates, it does not update). So a canvas is
fully described by (the token ids of every cache-writing call before it, the block's token ids),
and replaying those calls in order rebuilds the cache bit for bit -- Family B is bit-reproducible
(`results/ENV.md`). The bank is recorded from the fork's own sampler (`batch_sample`) rather
than a re-implementation: the sampler's `log` hands over each pass's type before its forward, as
in `calibrate_depth_v2.py`, and a pre-hook on `embed_tokens` sees the ids.

**The commit rule is the sampler's**: softmax in the logits' dtype (bf16), `x1_p > threshold`
strictly, plus the forced argmax, on logits shifted by one position (AR-converted: position i is
predicted from position i-1). Confidence for terciles, tau_eff and A3 is the fp32 max-probability
-- bf16 probabilities sit on a 0.0039 grid near 0.9, and quantiles of a grid are mostly ties.

**epsilon.** §2's epsilon is "the pooled KL of the static identity-kl:7 set", a Family A set.
Family B's grid found no accuracy-neutral static set: the deepest cosine identity cell whose CI
touched zero is identity:3 (-4.0 pt [-8.3, +0.3], `results/phase0.25/gate_B_cos.txt`); k = 6 is
-32.7. epsilon here is the pooled KL of that set, and every static cosine set k = 1..10 is
recorded so the reading can be redone at any other choice. Flagged as a deviation.

Standard rules on L = 28 (keep_first 1, keep_last 8) leave layers 1-19 and cap k at 10.

    python phase1_tok_v2.py --out results/phase1/tok_B --state /scratch1/.../tok_B.state.pt
"""
import argparse
import json
import os
import random
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import generation_functions                                             # noqa: E402
import profile_step_v2 as P                                             # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM            # noqa: E402
from dllm_skip.hook import install_skipping, uninstall_skipping, _find_layers  # noqa: E402
from dllm_skip.depth_schedule import select_skips                       # noqa: E402

MASK_ID = 151665
BUCKETS = [("(0.8, 1.0]", 0.8, 1.01), ("(0.6, 0.8]", 0.6, 0.8), ("(0.4, 0.6]", 0.4, 0.6),
           ("(0.2, 0.4]", 0.2, 0.4), ("[0.0, 0.2]", -0.01, 0.2)]
CLASSES = ["digit", "operator", "other"]


def bucket_of(r):
    for i, (_, lo, hi) in enumerate(BUCKETS):
        if lo < r <= hi:
            return i
    raise ValueError(r)


def token_class(tok, tid):
    """digit / operator / other, from the tokenizer (PREREG §1); as in v1/llada/phase1_map.py."""
    s = tok.decode([int(tid)]).strip()
    if not s:
        return "other"
    if any(c.isdigit() for c in s):
        return "digit"
    if all(not c.isalnum() for c in s):
        return "operator"
    return "other"


class Recorder:
    """Stands in for the sampler's `log` list. `append` is called with each pass's record just
    before its forward; the `embed_tokens` pre-hook then sees that forward's ids. Cache-writing
    calls are kept whole (they are what a replay needs); of the refinement passes, the first one
    in each r-bucket of each block is kept, as in Family A's bank."""

    def __init__(self):
        self.pending, self.calls, self.canvases, self.taken, self.problem = None, [], [], set(), -1

    def append(self, rec):
        self.pending = rec

    def __len__(self):
        return 0

    def hook(self, mod, args, kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        rec, self.pending = self.pending, None
        if rec is None or ids is None:
            return None
        if rec["cache_write"]:
            self.calls.append(ids.detach().cpu().clone())
            return None
        r = float((ids == MASK_ID).float().mean())
        b = bucket_of(r)
        if (rec["block"], b) not in self.taken:
            self.taken.add((rec["block"], b))
            self.canvases.append(dict(problem=self.problem, n_calls=len(self.calls),
                                      ids=ids.detach().cpu().clone(), r=r, bucket=b,
                                      block=rec["block"], step=rec["step"]))
        return None


class CacheReplay:
    """Rebuild the exact cache of a canvas by replaying its cache-writing calls. Canvases are
    visited in (problem, block) order, so the cache is extended rather than rebuilt when the
    next canvas is later in the same problem."""

    def __init__(self, model, calls, block_size):
        self.model, self.calls, self.bs = model, calls, block_size
        self.key, self.pkv = None, None

    @torch.no_grad()
    def get(self, c):
        p, n = c["problem"], c["n_calls"]
        if self.key is None or self.key[0] != p or self.key[1] > n:
            self.pkv, done = None, 0
        else:
            done = self.key[1]
        for i in range(done, n):
            out = self.model(input_ids=self.calls[p][i].to(self.model.device), use_cache=True,
                             past_key_values=self.pkv, update_past_key_values=True,
                             block_size=self.bs)
            self.pkv = out.past_key_values
        self.key = (p, n)
        return self.pkv


@torch.no_grad()
def block_logits(model, ids, pkv):
    lg = model(input_ids=ids, use_cache=True, past_key_values=pkv,
               update_past_key_values=False).logits
    return torch.cat([lg[:, :1], lg[:, :-1]], dim=1)        # the sampler's shift


def commit(lg, mask, thr):
    """The sampler's rule on shifted logits: bf16 softmax, x1_p > thr, forced argmax."""
    p = torch.softmax(lg, dim=-1)
    x1 = p.argmax(-1)
    x1_p = torch.gather(p, -1, x1.unsqueeze(-1)).squeeze(-1)
    x1_p = torch.where(mask, x1_p, torch.full_like(x1_p, -float("inf")))
    ti = x1_p > thr
    ti[torch.arange(lg.shape[0]), x1_p.argmax(-1)] = True
    return ti & mask


def kl_on(ref_logits, lg, m):
    p = torch.log_softmax(ref_logits[m].float(), dim=-1)
    q = torch.log_softmax(lg[m].float(), dim=-1)
    return float((p.exp() * (p - q)).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    ap.add_argument("--calib", default="/home1/hyunhoko/DLLM/results/phase0.25/calib_cosine_B.json",
                    help="Family B's cosine calibration: candidate pool and the static sets")
    ap.add_argument("--calib-ids", default="/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json",
                    help="the 128 calibration problems, shared with Family A")
    ap.add_argument("--out", required=True, help="prefix: writes <out>.npz and <out>.json")
    ap.add_argument("--state", default=None, help="checkpoint; default <out>.state.pt")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--n-problems", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--small-block-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--keep-first", type=int, default=1)
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--kmax", type=int, default=10, help="the standard-rules ceiling on L = 28")
    ap.add_argument("--n-cands", type=int, default=16)
    ap.add_argument("--greedy-canvases", type=int, default=100, help="per (bucket, half)")
    ap.add_argument("--eps-k", type=int, default=3,
                    help="epsilon = pooled KL of the static cosine identity set of this k")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: only the first N problems")
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model)
    cal = json.load(open(a.calib))
    L = cal["n_layers"]
    order = cal["global_order"]
    protected = set(range(a.keep_first)) | set(range(L - max(a.keep_last, 1), L))
    cands = [l for l in order if l not in protected][:a.n_cands]
    ext_pool = [l for l in range(L) if l not in protected]
    static_sets = {f"S{k}": sorted(select_skips(order, k, L, a.keep_first, a.keep_last, True))
                   for k in range(1, a.kmax + 1)}
    print(f"L={L}  static cosine sets: {static_sets}", flush=True)

    calib_ids = json.load(open(a.calib_ids))["calib_ids"][:a.n_problems]
    if a.limit:
        calib_ids = calib_ids[:a.limit]
    q, _ = P._gsm8k("train")
    prompts = []
    for i in calib_ids:                                   # the runner's and calibration's prompt
        text = f"Question: {q[i]}\nAnswer:".replace("Answer:", P.GSM8K_INSTRUCTION)
        prompts.append(tok.apply_chat_template([{"role": "user", "content": text}],
                                               add_generation_prompt=True, tokenize=False))

    state_path = a.state or (a.out + ".state.pt")
    st = torch.load(state_path, weights_only=False) if os.path.exists(state_path) else {}

    # ---- the bank, from the sampler itself, at full depth ----------------------------------
    if "bank" in st:
        bank, calls = st["bank"], st["calls"]
        print(f"resumed bank: {len(bank)} canvases", flush=True)
    else:
        rec = Recorder()
        inner = getattr(model, "model", model)
        h = inner.embed_tokens.register_forward_pre_hook(rec.hook, with_kwargs=True)
        calls = []
        t0 = time.perf_counter()
        for pi, text in enumerate(prompts):
            rec.problem, rec.calls, rec.taken = pi, [], set()
            ids = tok([text], return_tensors="pt").input_ids.to(dev)
            assert ids.shape[1] > a.block_size, "prompt shorter than a block: no prefill call"
            model.mdm_sample(ids, tokenizer=tok, block_size=a.block_size,
                             small_block_size=a.small_block_size,
                             max_new_tokens=a.max_new_tokens, mask_id=MASK_ID,
                             min_len=ids.shape[1], seq_len=torch.tensor([ids.shape[1]], device=dev),
                             use_block_cache=False, threshold=a.threshold,
                             schedule=None, controller=None, log=rec)
            calls.append(rec.calls)
            if (pi + 1) % 8 == 0:
                print(f"  bank: {pi+1}/{len(prompts)} problems, {len(rec.canvases)} canvases",
                      flush=True)
        h.remove()
        bank = rec.canvases
        print(f"bank: {len(bank)} canvases from {len(prompts)} problems "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        st["bank"], st["calls"] = bank, calls
        torch.save(st, state_path)
    half = {c["problem"]: c["problem"] % 2 for c in bank}

    ctrl = install_skipping(model)
    layers, _ = _find_layers(model)
    inner = getattr(model, "model", model)

    @torch.no_grad()
    def reference(c, replay, lens=False):
        pkv = replay.get(c)
        ids = c["ids"].to(dev)
        mask = ids == MASK_ID
        hs, hooks = {}, []
        if lens:
            for i, blk in enumerate(layers):
                hooks.append(blk.register_forward_hook(
                    lambda mod, args, out, i=i: hs.__setitem__(i, out[0] if isinstance(out, tuple) else out)))
        lg = block_logits(model, ids, pkv)
        for hk in hooks:
            hk.remove()
        ti = commit(lg, mask, a.threshold)
        return pkv, ids, lg, ti, mask, hs

    @torch.no_grad()
    def skip_pass(ids, keep, pkv):
        ctrl.arm(tuple(keep))
        lg = block_logits(model, ids, pkv)
        ctrl.arm(None)
        return lg

    @torch.no_grad()
    def crystallisation(hs, final_arg, pos):
        """Logit lens: position p is predicted from the hidden state at p - 1 (the shift). The
        last layer's output goes through the final norm like every other layer's, so its lens
        IS the model's logits."""
        W = model.lm_head.weight
        agree = torch.stack([torch.nn.functional.linear(inner.norm(hs[l][0, pos - 1]), W).argmax(-1)
                             == final_arg for l in range(len(layers))])
        out = torch.zeros(agree.shape[1], dtype=torch.long, device=agree.device)
        for l in range(len(layers) - 1, -1, -1):
            out[~agree[l] & (out == 0)] = l + 1
        return out

    # ---- A: split-half per-bucket greedy -----------------------------------------------------
    def greedy_for(canvases):
        chosen, trace = [], []
        while len(chosen) < a.kmax:
            def admissible(pool):
                return [l for l in pool if l not in chosen and not any(abs(l - x) == 1 for x in chosen)]
            pool, tag = admissible(cands), "registered"
            if not pool:
                pool, tag = admissible(ext_pool), "extended"
            if not pool:
                trace.append(dict(step=len(chosen) + 1, exhausted=True, set=sorted(chosen)))
                break
            tot, n = {l: 0.0 for l in pool}, 0
            replay = CacheReplay(model, calls, a.block_size)
            for c in canvases:
                pkv, ids, ref_lg, ti, _m, _hs = reference(c, replay)
                n += 1
                for l in pool:
                    keep = [j for j in range(L) if j not in chosen + [l]]
                    tot[l] += kl_on(ref_lg, skip_pass(ids, keep, pkv), ti)
            best = min(pool, key=lambda l: tot[l])
            chosen.append(best)
            trace.append(dict(step=len(chosen), chosen=best, kl=tot[best] / max(n, 1), pool=tag,
                              set=sorted(chosen), n_canvases=n,
                              all_scores={str(l): tot[l] / max(n, 1) for l in pool}))
        return trace

    greedy = dict(st.get("greedy", {}))
    for bi, (bname, _lo, _hi) in enumerate(BUCKETS):
        for hh in (0, 1):
            key = f"{bi}|{hh}"
            if key in greedy:
                print(f"  greedy {bname} half{hh}: already done", flush=True)
                continue
            idx = [i for i, c in enumerate(bank) if c["bucket"] == bi and half[c["problem"]] == hh]
            if not idx:
                continue
            pick = sorted(random.Random(1000 * bi + hh).sample(idx, min(a.greedy_canvases, len(idx))))
            t0 = time.perf_counter()
            tr = greedy_for([bank[i] for i in pick])
            greedy[key] = dict(bucket=bname, half=hh, canvas_idx=pick,
                               n_problems=len({bank[i]["problem"] for i in pick}), trace=tr)
            st["greedy"] = greedy
            torch.save(st, state_path)
            print(f"  greedy {bname} half{hh}: {[t.get('chosen') for t in tr]} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---- B: per-token records ------------------------------------------------------------------
    configs = [f"L{l}" for l in range(1, L - 1)]
    configs += list(static_sets)
    seq = {}
    for key, g in greedy.items():
        bi, hh = map(int, key.split("|"))
        seq[(bi, hh)] = [t["chosen"] for t in g["trace"] if "chosen" in t]
        configs += [f"G{bi}h{hh}k{k}" for k in range(1, len(seq[(bi, hh)]) + 1)]
    cid = {n: i for i, n in enumerate(configs)}

    def skipset(name, bucket):
        if name[0] == "L":
            return [int(name[1:])]
        if name[0] == "S":
            return static_sets[name]
        bi, rest = name[1:].split("h")
        hh, k = rest.split("k")
        return seq[(int(bi), int(hh))][:int(k)] if int(bi) == bucket else None

    recs = st.get("recs", [])
    start = len(recs)
    if start:
        print(f"resumed records at canvas {start}/{len(bank)}", flush=True)
    cls_cache = {}
    replay = CacheReplay(model, calls, a.block_size)
    t0 = time.perf_counter()
    for ci in range(start, len(bank)):
        c = bank[ci]
        pkv, ids, ref_lg, ti, maskb, hs = reference(c, replay, lens=True)
        m, mask = ti[0], maskb[0]
        pos = m.nonzero().squeeze(-1)
        n_full = int(m.sum())
        assert n_full >= 1 and int(pos.min()) >= 1, ci
        ref_arg = ref_lg[0].argmax(-1)
        cry = crystallisation(hs, ref_arg[pos], pos)
        pf = torch.softmax(ref_lg[0].float(), -1).max(-1).values          # fp32 confidence
        tids = ref_arg[pos].tolist()
        for t in tids:
            if t not in cls_cache:
                cls_cache[t] = CLASSES.index(token_class(tok, t))
        rec = dict(ci=ci, bucket=c["bucket"], half=half[c["problem"]], problem=c["problem"],
                   block=c["block"], step=c["step"], r=c["r"], n_mask=int(mask.sum()),
                   n_commit=n_full, conf=pf[pos].cpu(),
                   cls=torch.tensor([cls_cache[t] for t in tids], dtype=torch.int8),
                   cryst=cry.to(torch.int8).cpu(), cfg={}, mp_full=pf[mask].cpu())
        p = torch.log_softmax(ref_lg[0][pos].float(), -1)
        full_set = set(pos.tolist())
        for name in configs:
            sk = skipset(name, c["bucket"])
            if sk is None:
                continue
            lg = skip_pass(ids, [j for j in range(L) if j not in sk], pkv)
            q = torch.log_softmax(lg[0][pos].float(), -1)
            klv = (p.exp() * (p - q)).sum(-1)
            ag = lg[0][pos].argmax(-1) == ref_arg[pos]
            mp = torch.softmax(lg[0][mask].float(), -1).max(-1).values
            mpos = mask.nonzero().squeeze(-1)
            o = torch.argsort(mp, descending=True)
            top = set(mpos[o[:n_full]].tolist())
            n09 = int(commit(lg, maskb, a.threshold).sum())               # the sampler's rule
            rec["cfg"][cid[name]] = dict(
                kl=klv.cpu(), agree=ag.cpu(), n09=n09, tau_eff=float(mp[o[n_full - 1]]),
                jac=len(top & full_set) / len(top | full_set),
                maxprob=mp.cpu() if name[0] == "S" else None)
        recs.append(rec)
        if (ci + 1) % 50 == 0:
            print(f"  records: {ci+1}/{len(bank)} "
                  f"({(time.perf_counter()-t0)/(ci+1-start):.2f}s/canvas)", flush=True)
        if (ci + 1) % a.save_every == 0:
            st["recs"] = recs
            torch.save(st, state_path)
    st["recs"] = recs
    torch.save(st, state_path)
    uninstall_skipping(model)

    # ---- flatten: the layout of v1/llada/phase1_tok.py ---------------------------------------
    C, T, N = len(configs), sum(len(r["conf"]) for r in recs), len(recs)
    kl_a = np.full((C, T), np.nan, np.float32)
    ag_a = np.full((C, T), -1, np.int8)
    n09 = np.full((C, N), -1, np.int32)
    taue = np.full((C, N), np.nan, np.float32)
    jac = np.full((C, N), np.nan, np.float32)
    mp_vals, mp_canvas, mp_cfg = [], [], []
    off = 0
    for j, r in enumerate(recs):
        nt = len(r["conf"])
        mp_vals.append(r["mp_full"].numpy().astype(np.float16))
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
        jac=jac,
        tok_canvas=np.concatenate([np.full(len(r["conf"]), j, np.int32) for j, r in enumerate(recs)]),
        tok_conf=np.concatenate([r["conf"].numpy() for r in recs]).astype(np.float32),
        tok_cls=np.concatenate([r["cls"].numpy() for r in recs]),
        tok_cryst=np.concatenate([r["cryst"].numpy() for r in recs]),
        c_bank_idx=cv("ci", np.int32), c_bucket=cv("bucket", np.int8), c_half=cv("half", np.int8),
        c_problem=cv("problem", np.int16), c_block=cv("block", np.int8),
        c_step=cv("step", np.int16), c_r=cv("r", np.float32), c_n_mask=cv("n_mask", np.int16),
        c_n_commit=cv("n_commit", np.int16),
        mp_vals=np.concatenate(mp_vals), mp_canvas=np.concatenate(mp_canvas),
        mp_cfg=np.concatenate(mp_cfg))
    json.dump(dict(model=a.model, L=L, n_canvases=N, n_tokens=T, args=vars(a),
                   buckets=[b[0] for b in BUCKETS], classes=CLASSES, candidates=cands,
                   extended_pool=ext_pool, static_sets=static_sets, eps_config=f"S{a.eps_k}",
                   static_label="static cosine identity sets (select_skips on calib_cosine_B)",
                   configs=configs, greedy=greedy), open(a.out + ".json", "w"), indent=1)
    print(f"written {a.out}.npz / .json  ({N} canvases, {T} committed tokens, {C} configs)")


if __name__ == "__main__":
    main()
