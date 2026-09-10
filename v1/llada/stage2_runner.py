"""Phase 0.25 Stage 2 runner, Family A — `results/phase0.25/PREREG.md` §2-§5.

One process loads the model once and walks a list of cells; each cell writes
`<out>/<mode>/<L_eq>/summary.json` and `steps.jsonl` and is skipped if its summary already
exists, so a preempted job resumes instead of restarting.

Why not lm-eval. Every Stage 2 outcome is a **paired per-problem** comparison against the
`full` reference on the same problems (§5), on E — 300 GSM8K-*train* problems fixed in
`ids.json` — and the gate reads per-problem wall-clock. lm-eval runs task splits and reports
neither. Scoring reproduces lm-eval's own flexible-extract filter so the numbers stay
comparable with `../phase0/` (last regex match of `(-?[$0-9.,]{2,})|(-?[0-9]+)`).

Per-pass confidence comes from the sampler's own tensor via
`get_transfer_index(..., return_confidence=True)` (`01` §1, decided 2026-09-09): that softmax
is already inside the timed baseline, so the record costs nothing. Reductions happen on the
GPU and the scalars move off once per problem, never inside the pass loop.

The `full` reference runs twice, first and last in the cell list, and the two wall-clocks
bound how much the node drifted under us; every speedup is reported against the mean.
"""
import argparse, json, os, re, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import generate as G                                                   # noqa: E402
import profile_step as P                                               # noqa: E402
from model.modeling_llada import LLaDAModelLM                          # noqa: E402
from transformers import AutoTokenizer                                 # noqa: E402
from dllm_skip.hook import install_skipping, uninstall_skipping, MODES  # noqa: E402
from dllm_skip.depth_schedule import (DepthSchedule, RegimeSchedule, StepState,
                                      select_skips, leq_share)  # noqa: E402

MASK_ID = 126336
LAYERS = 32
ANS = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def flexible_extract(text):
    m = ANS.findall(text)
    if not m:
        return None
    last = [g for g in m[-1] if g][-1]
    return last.replace("$", "").replace(",", "").rstrip(".")


def gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def build(tok, ids, n_shot, split="train"):
    q, a = P._gsm8k(split)
    shots = "".join(P.FEWSHOT_TEMPLATE.format(q=q[i], a=a[i]) for i in range(n_shot))
    return [tok.apply_chat_template(
        [{"role": "user", "content": shots + f"Question: {q[i]}\nAnswer:"}],
        add_generation_prompt=True, tokenize=False) for i in ids], [gold(a[i]) for i in ids]


class PassLog:
    """Per-pass confidence record.

    `generate.py` already appends each pass record to the `log` list, and `sink.add` receives
    that same dict object -- so this must NOT append it again, or the record count doubles and
    the statistics land on the wrong passes. It keeps (record, gpu_stats) pairs and fills the
    records in place at `drain()`, which is the only sync and happens once per problem.
    """

    def __init__(self, threshold):
        self._pending, self.thr = [], threshold

    def add(self, rec, confidence, mask, committed):
        if confidence is None:
            return
        c = confidence[mask]
        stats = (torch.stack([c.mean(), c.min(), (c >= self.thr).sum().to(c.dtype),
                              committed.sum().to(c.dtype)])
                 if c.numel() else None)
        self._pending.append((rec, stats))

    def drain(self):
        vals = [None if t is None else t.tolist() for _, t in self._pending]   # one sync
        for (rec, _), v in zip(self._pending, vals):
            if v is not None:
                rec["conf_mean"], rec["conf_min"], rec["n_ge_tau"], rec["committed"] = v
        self._pending = []


def run_cell(model, tok, prompts, golds, ids, sched, ctrl, mode, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary_p = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_p):
        print(f">>> SKIP {out_dir}", flush=True)
        return json.load(open(summary_p))
    if ctrl is not None:
        ctrl.mode = mode
    per, steps_f = [], open(os.path.join(out_dir, "steps.jsonl"), "w")
    plog = PassLog(args.threshold)
    torch.cuda.synchronize()
    for n, (text, g, pid) in enumerate(zip(prompts, golds, ids)):
        inp = tok(text, return_tensors="pt").input_ids.to(model.device)
        log = []
        if ctrl is not None:
            ctrl.new_block()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            out, st = G.generate_with_dual_cache(
                model, inp, steps=args.steps, gen_length=args.gen_length,
                block_length=args.block_length, temperature=0.0, remasking="low_confidence",
                threshold=args.threshold, tau_r=args.tau_r, schedule=sched, controller=ctrl,
                log=log, sink=plog)
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
        gen = tok.decode(out[0, inp.shape[1]:], skip_special_tokens=True)
        pred = flexible_extract(gen)
        try:
            ok = pred is not None and abs(float(pred) - float(g)) < 1e-6
        except ValueError:
            ok = False
        per.append(dict(problem=int(pid), correct=int(ok), wall_s=wall, nfe=int(st),
                        layer_steps=int(st.layer_steps), full_layer_steps=int(st.full_layer_steps),
                        pred=pred, gold=g))
        plog.drain()                    # one sync per problem, never inside the pass loop
        for r in log:
            r["problem"] = int(pid)
            steps_f.write(json.dumps(r) + "\n")
        if (n + 1) % 50 == 0:
            print(f"    {n+1}/{len(prompts)}", flush=True)
    steps_f.close()
    acc = sum(p["correct"] for p in per) / len(per)
    summ = dict(mode=mode, n=len(per), accuracy=acc,
                wall_s=sum(p["wall_s"] for p in per),
                nfe=sum(p["nfe"] for p in per),
                layer_steps=sum(p["layer_steps"] for p in per),
                full_layer_steps=sum(p["full_layer_steps"] for p in per),
                budget=None if sched is None else sched.budget,
                # the SET the cell skips, read from the base schedule. A RegimeSchedule
                # returns every layer on a protected pass, so probing it at one mask ratio
                # would record an empty set for whichever regime does not skip there.
                skipped=None if sched is None else sorted(
                    set(range(sched.n_layers))
                    - set(getattr(sched, "base", sched).active_layers(StepState(0.5)))),
                # PREREG 0.4 §2: the regime and its boundary travel with the cell. A regime
                # cell and the static cell it is compared with differ in exactly one field, and
                # this is it -- reading them apart from the directory name would be a guess.
                regime=getattr(sched, "regime", "static"),
                r_star=getattr(sched, "r_star", None),
                tau_w=args.threshold, tau_r=args.tau_r,
                args=vars(args), per_problem=per)
    # Raw layer count, then the byte-weighted L_eq ratio the gate axis is defined in:
    # a skipped `no-attn` layer still runs its FFN, so counting it as zero understates the
    # cost badly (0.10 instead of ~0.87 in Family B).
    fls = summ["full_layer_steps"]
    summ["depth_ratio_layers"] = summ["layer_steps"] / fls if fls else float("nan")
    share = leq_share(mode, LAYERS)
    skipped_layer_steps = fls - summ["layer_steps"]
    summ["leq_removed"] = skipped_layer_steps * share
    summ["depth_ratio"] = 1.0 - (summ["leq_removed"] / fls) if fls else float("nan")
    json.dump(summ, open(summary_p, "w"), indent=1)
    print(f">>> DONE {out_dir}  acc={acc:.4f}  wall={summ['wall_s']:.1f}s  "
          f"nfe={summ['nfe']}  depth={summ['depth_ratio']:.4f}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--ids", default="/home1/hyunhoko/DLLM/results/phase0.25/ids.json")
    ap.add_argument("--split", default=None, choices=["train", "test"],
                    help="which GSM8K split the ids index. Every phase up to 0.3's n = 300 cells "
                         "uses `train` (E is drawn from it); PREREG 0.3 §3(c) confirms on the "
                         "test split at n = 1319. Read from the ids file's own `split` field when "
                         "omitted; if both are given they must agree.")
    ap.add_argument("--calib", default="/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json")
    ap.add_argument("--kl-json", default="/home1/hyunhoko/DLLM/results/phase0.25/klgreedy_A.json",
                    help="comma-separated KL-greedy search outputs; each is indexed by the k it "
                         "records, and its own `rules` (keep_first/keep_last/no_consecutive) "
                         "travel with it -- PREREG 0.4 §2's k = 16 set is searched under relaxed "
                         "rules and must not be rebuilt under the standard ones")
    ap.add_argument("--regimes", default="static",
                    help="comma-separated: early | late | static (PREREG 0.4 §2). `early` skips "
                         "only where the block mask ratio r > r*, `late` only where r <= r*.")
    ap.add_argument("--r-star", type=float, default=None,
                    help="the regime boundary, measured in Phase 0.35. Required unless every "
                         "regime is `static`.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", required=True, help="mode:k,mode:k,... ; 'full' for the reference")
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="tau_w: the threshold on cache-writing passes. Held at 0.9 in every "
                         "Phase 0.3 cell -- those passes are full depth everywhere, their "
                         "confidence is not deflated, and the baseline and depth rows must move "
                         "the same parameter (PREREG 0.3 §2).")
    ap.add_argument("--tau-r", type=float, default=None,
                    help="the threshold on refinement passes. None = use --threshold everywhere "
                         "(every phase before 0.3). A regime row applies it only on the passes "
                         "its regime actually skips.")
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all of E")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    ids_obj = json.load(open(a.ids))
    E = ids_obj["E"]
    # results/README rule 4: the split is a registered input, so it is asserted and logged rather
    # than inferred. Reading test-split indices against the train split would silently score the
    # wrong 1319 problems and every number downstream would be wrong but plausible.
    file_split = ids_obj.get("split")
    split = a.split or file_split or "train"
    if a.split and file_split and a.split != file_split:
        raise SystemExit(f"--split {a.split} but {a.ids} declares split={file_split}")
    a.split = split
    if a.limit:
        E = E[:a.limit]
    prompts, golds = build(tok, E, a.n_shot, split)
    cal = json.load(open(a.calib))
    L = cal["n_layers"]
    order = cal["global_order"]
    ctrl = install_skipping(model)

    # results/README rule 4: every registered input is asserted present and logged at start-up;
    # a silent fallback is what put Phase 0.25's Family B grid on the excluded skip set.
    regimes = [r.strip() for r in a.regimes.split(",") if r.strip()]
    for r in regimes:
        assert r in ("early", "late", "static"), f"unknown regime {r!r}"
    if any(r != "static" for r in regimes):
        assert a.r_star is not None, "--r-star is required for a non-static regime (Phase 0.35)"
    kl_sets = {}
    for path in [q.strip() for q in a.kl_json.split(",") if q.strip()]:
        if not os.path.exists(path):
            raise SystemExit(f"registered input missing: {path}")
        j = json.load(open(path))
        rules = j.get("rules", dict(keep_first=1, keep_last=a.keep_last, no_consecutive=True))
        kl_sets[int(j["k"])] = (sorted(j["kl_greedy_set"]), rules,
                                j.get("label", "standard"), path)
    print(f"E: {len(E)} problems from GSM8K-{split} ({a.ids}) | L={L} | regimes: {regimes} | "
          f"r* = {a.r_star} | tau_w = {a.threshold} | tau_r = {a.tau_r} | cells: {a.cells}",
          flush=True)
    for k, (st, rules, label, path) in sorted(kl_sets.items()):
        print(f"  KL set k={k:<3} {st}  rules={rules}  label={label!r}  <- {path}", flush=True)

    # PREREG 0.3 sweeps tau_r, so the cell path carries it; without this a sweep would write
    # every tau into the same directory and resume would skip the rest of it.
    if a.tau_r is not None:
        a.out = os.path.join(a.out, f"tau{a.tau_r:g}")
    for spec in a.cells.split(","):
        spec = spec.strip()
        if spec == "full":
            # A k = 0 schedule, not `None`. With no schedule the sampler skips the whole
            # instrumentation path -- no StepState, no per-pass `.item()` -- so the reference
            # would not pay the measurement cost the cells pay, and every measured S would be
            # biased in the cells' favour... no, against them. The gate reads wall-clock
            # (PREREG section 6), so the reference must run the identical code path at full
            # depth. It also makes layer_steps/full_layer_steps well defined for this row.
            full_sched = DepthSchedule.static(L, [order], 0, keep_first=1,
                                              keep_last=a.keep_last, no_consecutive=True)
            # a full-depth run has no regime, so it is written once, outside the regime tree
            run_cell(model, tok, prompts, golds, E, full_sched, ctrl, "identity", a,
                     os.path.join(a.out, "full", "0"))
            continue
        mode, k = spec.split(":")
        k = int(k)
        # `identity-kl` takes its skip set from the KL-greedy search on S (PREREG section 3)
        # instead of the cosine ranking -- the one cell that asks whether selection quality,
        # not budget size, moves the pass count.
        if mode == "identity-kl":
            if k not in kl_sets:
                raise SystemExit(f"no KL-greedy set for k={k} in --kl-json ({sorted(kl_sets)})")
            klset, rules, label, _src = kl_sets[k]
            assert len(klset) == k, f"kl set has {len(klset)} layers, cell asks {k}"
            kl_order = klset + [l for l in range(L) if l not in klset]
            # the set's own rules, not this run's flags: the k = 16 set is searched under
            # relaxed ones and rebuilding it under the standard ones would silently truncate it
            base = DepthSchedule.static(L, [kl_order], k, keep_first=int(rules["keep_first"]),
                                        keep_last=int(rules["keep_last"]),
                                        no_consecutive=bool(rules["no_consecutive"]))
            got = sorted(set(range(L)) - set(base.active_layers(StepState(0.5))))
            assert got == klset, f"schedule picked {got}, not the KL set {klset} ({label})"
            ctrl.reuse_m = None
            for regime in regimes:
                sched = base if regime == "static" else \
                    RegimeSchedule(base=base, regime=regime, r_star=a.r_star)
                run_cell(model, tok, prompts, golds, E, sched, ctrl, "identity", a,
                         os.path.join(a.out, regime, "identity-kl", str(k)))
            continue
        # `reuse2` / `reuse4` / `reuse` select the refresh interval m (reuse = inf)
        m = None
        if mode.startswith("reuse") and mode != "reuse":
            m = int(mode[len("reuse"):]); mode = "reuse"
        ctrl.reuse_m = m
        assert mode in MODES, mode
        nc = mode in ("identity", "reuse")           # non-adjacency only for whole-layer modes
        base = DepthSchedule.static(L, [order], k, keep_first=1,
                                    keep_last=a.keep_last if nc else 0, no_consecutive=nc)
        name = mode if m is None else f"{mode}{m}"
        for regime in regimes:
            sched = base if regime == "static" else \
                RegimeSchedule(base=base, regime=regime, r_star=a.r_star)
            run_cell(model, tok, prompts, golds, E, sched, ctrl, mode, a,
                     os.path.join(a.out, regime, name, str(k)))
    uninstall_skipping(model)


if __name__ == "__main__":
    main()
