"""Check M3 of Fast-dLLM v2 7B: the KL-greedy layer set (Phase 3 amendments A14, change 14 and A17, change 1).

The rule of `v1/llada/kl_greedy.py` (the set of LLaDA-8B-Instruct), on the sampler of this model:
- S: the 24 GSM8K train problems of `results/phase0.25/ids.json`, with the GSM8K prompt of the cells of this model
  (A7: 0-shot, the chat template, the boxed-answer instruction of `v2/eval.py`; A17, change 1, question 2).
- The bank. The sampler itself (`generation_functions.batch_sample`, full depth, the settings of A7, tau 0.9) runs on S
  in order. A wrapper of `model.forward` and a sink record each refinement pass; the sampler does not change. In each
  block the refinement passes count from 1. At each fourth pass (stride 4; each refinement pass commits a token), the
  bank keeps the 32 tokens of the block (the input of the pass), the logits of its small block after the shift of one
  position that the sampler applies, the positions that the pass commits, and the cache of the block. The canvases of
  one block share one copy of the cache. The bank stops at the end of the block in which its count reaches 96.
- The cache (A17, change 1, question 1, option (a)): the pass with fewer layers uses the cache of the sampler. The
  attention is block-causal and a refinement pass does not write the cache (`modeling.py` of revision 0661abf: with
  update_past_key_values=False the layer joins the cache and the block with torch.cat). So this is the cache that the
  rule of LLaDA-8B-Instruct builds again from the canvas, and the set with the reference pass on the same cache as the
  pass with fewer layers (A14, change 14, rule 5) is this set.
- The score: the mean over the canvases of the mean KL(full || pruned) over the committed positions, in float32.
- Before the search, the KL with no layer removed. It must be 0; if not, the script writes the bank facts and stops
  with no set (A17, change 1).
- The pool rules of A4, change 3 for L = 28: keep layer 0 and the last 7, no two adjacent, k = 6. They go into the set
  file, and the runner checks them (A14, change 14, rule 7). The file also records the count of S problems in the bank
  and the second-best score at each step (rule 4).
- An output with the same arguments and fork commit is skipped, so a rerun or a stop is harmless. The file is written
  whole.
"""
import argparse, copy, json, os, sys, time, types
from math import comb
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import generation_functions                                             # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "v1", "llada")))
import phase3_records as R                                              # noqa: E402
import profile_step_v2 as P                                             # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM            # noqa: E402
from dllm_skip.hook import install_skipping, uninstall_skipping         # noqa: E402
from dllm_skip.depth_schedule import select_skips, max_skippable        # noqa: E402

MASK_ID = 151665
A7 = dict(block_size=32, small_block_size=8, max_new_tokens=2048)


class StopBank(Exception):
    """The bank is full: the block in which the count reached the limit has ended."""


class Recorder:
    """The wrapper of model.forward and the sink of batch_sample for one bank."""

    def __init__(self, model, a):
        self.model, self.a = model, a
        self.inner = model.forward
        self.bank, self.caches = [], []
        self.problem = None
        self.block = None
        self.i = 0                      # refinement passes in the block, from 1
        self.cache_id = None            # the copy of the cache of the block in self.caches
        self.pending = None

    def forward(self, *args, **kw):
        upd = kw.get("update_past_key_values", False)
        if upd and len(self.bank) >= self.a.max_canvases:
            raise StopBank()            # a cache-writing pass: the block before it has ended
        out = self.inner(*args, **kw)
        if not upd:
            self.pending = (kw["input_ids"].clone(), out.logits, kw.get("past_key_values"))
        return out

    # the sink interface of batch_sample (DepthB calls add after each pass, then commits)
    def add(self, rec, conf, mask, committed):
        if rec["kind"] != "refine":
            return
        if rec["block"] != self.block:
            self.block, self.i, self.cache_id = rec["block"], 0, None
        self.i += 1
        self.take = self.i % self.a.stride == 0 and bool(committed.any())
        self.committed = committed.clone()

    def commits(self, rec, tokens, committed, pos0):
        if rec["kind"] != "refine" or not self.take:
            return
        x, logits, pkv = self.pending
        sbs = pos0 - rec["block"] * self.a.block_size        # start of the small block in the block
        if self.cache_id is None:
            self.caches.append(copy.deepcopy(pkv))
            self.cache_id = len(self.caches) - 1
        self.bank.append(dict(problem=self.problem, block=rec["block"], step=rec["step"], sbs=sbs, x=x,
                              cache=self.cache_id, ref=small_block(logits, sbs, self.a).detach().clone(),
                              committed=self.committed))


def small_block(logits, sbs, a):
    """The logits of the small block after the shift of one position, as batch_sample takes them."""
    lg = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
    return lg[:, sbs:sbs + a.small_block_size]


@torch.no_grad()
def build_bank(model, tok, prompts, S, a):
    rec = Recorder(model, a)
    model.forward = rec.forward
    n_run = 0
    try:
        for i, text in zip(S, prompts):
            if len(rec.bank) >= a.max_canvases:
                break
            rec.problem, rec.block = i, None
            n_run += 1
            ids_t = tok([text], return_tensors="pt").input_ids.to(model.device)
            L = ids_t.shape[1]
            try:
                model.mdm_sample(ids_t, tokenizer=tok, block_size=a.block_size, small_block_size=a.small_block_size,
                                 max_new_tokens=a.max_new_tokens, mask_id=MASK_ID, min_len=L,
                                 seq_len=torch.tensor([L], device=model.device), use_block_cache=False,
                                 threshold=a.threshold, sink=rec)
            except StopBank:
                break
    finally:
        model.forward = rec.inner
    return rec.bank, rec.caches, n_run


@torch.no_grad()
def mean_kl(model, ctrl, bank, caches, keep, a):
    """Mean KL(full || pruned) over the committed positions of each canvas, on the cache of the sampler."""
    tot, n = 0.0, 0
    for b in bank:
        ctrl.arm(keep)                  # single-shot: the arm applies to the next pass only
        lg = model.forward(input_ids=b["x"], use_cache=True, past_key_values=caches[b["cache"]],
                           update_past_key_values=False).logits
        m = b["committed"]
        p = F.log_softmax(b["ref"][m].float(), dim=-1)
        q = F.log_softmax(small_block(lg, b["sbs"], a)[m].float(), dim=-1)
        tot += float((p.exp() * (p - q)).sum(-1).mean())
        n += 1
    return tot / max(n, 1)


def n_admissible(L, k, keep_first, keep_last, no_consecutive):
    n = L - keep_first - max(keep_last, 1)
    if k > n:
        return 0
    return comb(n - k + 1, k) if no_consecutive else comb(n, k)


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    ap.add_argument("--model-revision", required=True)
    ap.add_argument("--ids", required=True, help="results/phase0.25/ids.json (its S)")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--small-block-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-canvases", type=int, default=96)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", action="store_true",
                    help="a test, not the check: stop after the KL with no layer removed (use a small --max-canvases)")
    a = ap.parse_args()
    got = dict(block_size=a.block_size, small_block_size=a.small_block_size, max_new_tokens=a.max_new_tokens)
    if got != A7:
        raise SystemExit(f"settings {got}, not those of A7 ({A7})")
    run = R.run_record()
    args = {k: v for k, v in vars(a).items() if k != "out"}
    if os.path.exists(a.out):
        old = json.load(open(a.out))
        if old.get("args") == args and old.get("run", {}).get("fork_commit") == run["fork_commit"]:
            print(f"SKIP {a.out} (same arguments and fork commit)", flush=True)
            return
    print(f"run: {run}", flush=True)

    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.model_revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.model_revision, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16).to(dev).eval()
    model.mdm_sample = types.MethodType(generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model)
    # the skipping wrappers are in place for the bank too (armed with no skip), so the reference pass and the scored
    # passes go through the same modules
    ctrl = install_skipping(model)          # mode identity: a skipped layer hands on its input
    L = ctrl.n_layers

    S = json.load(open(a.ids))["S"]
    q, _ = P._gsm8k("train")
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": f"Question: {q[i]}\nAnswer:".replace("Answer:", P.GSM8K_INSTRUCTION)}],
        add_generation_prompt=True, tokenize=False) for i in S]
    t0 = time.perf_counter()
    bank, caches, n_run = build_bank(model, tok, prompts, S, a)
    in_bank = sorted({b["problem"] for b in bank})
    print(f"bank: {len(bank)} canvases, {len(caches)} block caches, from {len(in_bank)} S problems "
          f"(the sampler ran on {n_run} of {len(S)}); {time.perf_counter() - t0:.0f} s", flush=True)

    keep_first, keep_last, nc = 1, round(L / 4), True
    k = round(0.22 * L)
    ceiling = max_skippable(L, keep_first, keep_last, nc)
    assert k <= ceiling, (k, ceiling)
    res = dict(model=a.model, model_revision=a.model_revision, model_snapshot=R.snapshot(a.model, a.model_revision),
               L=L, k=k, rules=dict(keep_first=keep_first, keep_last=keep_last, no_consecutive=nc, ceiling=ceiling),
               selection_set="S of Phase 0.25 (24 GSM8K train problems)", n_S=len(S), n_S_run=n_run,
               n_S_in_bank=len(in_bank), S_in_bank=in_bank, n_canvases=len(bank), n_block_caches=len(caches),
               canvases=[dict(problem=b["problem"], block=b["block"], step=b["step"], small_block_start=b["sbs"],
                              committed=int(b["committed"].sum())) for b in bank],
               cache="the cache of the sampler (A17, change 1, option (a)); also the set of A14, change 14, rule 5",
               prompt="GSM8K prompt of the cells of this model (A7; A17, change 1, option (a))",
               args=args, run=run)

    kl0 = mean_kl(model, ctrl, bank, caches, None, a)
    res["kl_no_removal"] = kl0
    print(f"KL with no layer removed: {kl0!r}", flush=True)
    if kl0 != 0.0:
        res.update(stopped="the KL with no layer removed is not 0 (A17, change 1)", kl_greedy_set=None)
        write_json(res, a.out)
        raise SystemExit(f"STOP: the KL with no layer removed is {kl0!r}, not 0; {a.out} has no set")
    if a.test:
        k1 = [j for j in range(L) if j not in (1, 3)]
        res.update(test=True, kl_greedy_set=None, kl_without_1_3=mean_kl(model, ctrl, bank, caches, k1, a))
        write_json(res, a.out)
        print(f"TEST: KL without layers 1 and 3 {res['kl_without_1_3']!r}; {a.out} (no set)", flush=True)
        return

    chosen, trace, last_scores, exhausted = [], [], {}, False
    protected = set(range(keep_first)) | set(range(L - max(keep_last, 1), L))
    for step in range(k):
        cands = [l for l in range(L) if l not in protected and l not in chosen
                 and not any(abs(l - c) == 1 for c in chosen)]
        if not cands:
            exhausted = True
            print(f"  greedy exhausted at {len(chosen)} of {k}; the ranked feasible set follows", flush=True)
            break
        scores = {}
        for l in cands:
            keep = [j for j in range(L) if j not in chosen + [l]]
            scores[l] = mean_kl(model, ctrl, bank, caches, keep, a)
        last_scores.update(scores)
        order = sorted(scores, key=scores.get)
        best = order[0]
        second = order[1] if len(order) > 1 else None
        chosen.append(best)
        trace.append(dict(step=step + 1, chosen=best, kl=scores[best], second=second,
                          kl_second=scores[second] if second is not None else None,
                          margin=(scores[second] - scores[best]) / scores[best]
                          if second is not None and scores[best] > 0 else None,
                          all_scores={str(x): v for x, v in sorted(scores.items())}))
        print(f"  step {step + 1}: +layer {best:2d} KL={scores[best]:.6f}; second {second} "
              f"KL={trace[-1]['kl_second']}; set={sorted(chosen)}", flush=True)
    uninstall_skipping(model)
    greedy_prefix = list(chosen)
    if exhausted or len(chosen) < k:
        rest = sorted((l for l in range(L) if l not in chosen and l in last_scores), key=lambda l: last_scores[l])
        tail = [l for l in range(L) if l not in chosen and l not in rest]
        chosen = list(select_skips(chosen + rest + tail, k, L, keep_first, keep_last, nc))
        print(f"  fallback set: {sorted(chosen)}", flush=True)
    res.update(kl_greedy_set=sorted(chosen), greedy_exhausted=bool(exhausted), greedy_prefix=sorted(greedy_prefix),
               n_admissible_sets=n_admissible(L, k, keep_first, keep_last, nc), trace=trace,
               seconds=round(time.perf_counter() - t0, 1), end=R.now())
    write_json(res, a.out)
    print(f"\n  KL-greedy set: {sorted(chosen)}\n  written {a.out}", flush=True)


if __name__ == "__main__":
    main()
