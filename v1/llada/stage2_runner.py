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
from model.modeling_llada import LLaDAModelLM, LLaDABlock, set_pad_mask  # noqa: E402
from transformers import AutoTokenizer                                 # noqa: E402
from dllm_skip.hook import install_skipping, uninstall_skipping, MODES  # noqa: E402
from dllm_skip.depth_schedule import (DepthSchedule, RegimeSchedule, StepState,
                                      select_skips, leq_share)  # noqa: E402

MASK_ID = 126336
LAYERS = 32
CODE = {}           # Phase 2 P4: the lm-eval task object of a code task (kept out of args, which go to summary.json)
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

    `n_ge_tau` counts the positions >= tau_w on every pass, as before. Phase 2 (amendment A3) adds
    `thr` and `n_ge_thr` on a pass whose applied threshold differs from tau_w (a tau_r cell), so
    the floor rule of that pass can be read; and the sampler now also records cache-writing passes.
    Amendment A6 adds `n_ge_09`, the positions >= 0.9, on every pass. With `rows=True` (a batch of
    more than one problem) each pass also gets, for each row, the masked positions of the block
    before the pass (`rows_masked`), the commits (`rows_committed`) and the positions >= 0.9
    (`rows_n_ge_09`); the pooled fields cover the whole batch.
    """

    def __init__(self, threshold, rows=False):
        self._pending, self.thr, self.rows = [], threshold, rows

    def add(self, rec, confidence, mask, committed, thr=None):
        if confidence is None:
            return
        c = confidence[mask]
        t = self.thr if thr is None else thr
        stats = (torch.stack([c.mean(), c.min(), (c >= self.thr).sum().to(c.dtype),
                              committed.sum().to(c.dtype), (c >= t).sum().to(c.dtype),
                              (c >= 0.9).sum().to(c.dtype)])
                 if c.numel() else None)
        rows = (torch.stack([mask.sum(-1), (committed & mask).sum(-1), ((confidence >= 0.9) & mask).sum(-1)])
                if self.rows else None)
        self._pending.append((rec, stats, t, rows))

    def drain(self):
        vals = [(None if s is None else s.tolist(), None if r is None else r.tolist())
                for _, s, _, r in self._pending]                                   # one sync
        for (rec, _, t, _), (v, rw) in zip(self._pending, vals):
            if v is not None:
                rec["conf_mean"], rec["conf_min"], rec["n_ge_tau"], rec["committed"] = v[:4]
                rec["n_ge_09"] = v[5]
                if t != self.thr:
                    rec["thr"], rec["n_ge_thr"] = t, v[4]
            if rw is not None:
                rec["rows_masked"], rec["rows_committed"], rec["rows_n_ge_09"] = rw
        self._pending = []


def batch_groups(enc, ids, B):
    """Phase 2 P5: the batches of a cell, as lists of indices into the prompts.

    B = 1: one problem per batch, in the original order (the old behaviour). B > 1: the problems sorted by prompt
    length, then by problem id, and cut into consecutive batches of B. So every cell of a run has the same batches,
    and a batch has little padding.
    """
    if B == 1:
        return [[i] for i in range(len(enc))]
    order = sorted(range(len(enc)), key=lambda i: (len(enc[i]), ids[i]))
    return [order[k:k + B] for k in range(0, len(enc), B)]


def compression_record(args):
    """docs/plan/03_protocol.md section 6: kind, bits, method, revision, SoloQ environment."""
    soloq_env = {k: v for k, v in os.environ.items() if k.startswith("SOLOQ_") and k != "SOLOQ_PATH"}
    if getattr(args, "gptq", None):
        rec = dict(kind="weights", bits=getattr(args, "gptq_bits", None),
                   method=f"gptq expanded ({args.gptq_kernel})", revision=args.gptq)
    elif getattr(args, "rtn_bits", None):
        errs = list(args.rtn_errors.values())
        rec = dict(kind="weights", bits=args.rtn_bits, method="rtn symmetric group 128", revision=None,
                   mean_rel_weight_error=sum(errs) / len(errs))
    elif soloq_env:
        rec = dict(kind="weights+activations", bits=None, method="soloq", revision=None)
    else:
        rec = dict(kind="none", bits=None, method=None, revision=None)
    rec["soloq_env"] = soloq_env
    if getattr(args, "gptq", None) and args.gptq_kernel == "marlin":
        rec["marlin_row_max"] = args.marlin_row_max
        rec["cache_write_passes"] = "bf16 copy of the expanded weights (calls with more rows than marlin_row_max)"
    return rec


def marlin_record(args, tot):
    """Phase 2 P5 (gate G1'.4): the Marlin calls of a cell against its passes. Every block linear runs once per
    pass, so each refinement pass used Marlin and each cache-writing pass used the bf16 copy exactly when the calls
    equal the passes times the number of Marlin linears."""
    if not (getattr(args, "gptq", None) and args.gptq_kernel == "marlin"):
        return None
    nl = args.marlin_linears
    return dict(row_max=args.marlin_row_max, linears=nl, calls=dict(marlin=tot["marlin"], bf16=tot["bf16"]),
                refine_passes=tot["refine_passes"], cache_write_passes=tot["cache_write_passes"],
                every_refine_pass_marlin=tot["marlin"] == nl * tot["refine_passes"],
                every_cache_write_pass_bf16=tot["bf16"] == nl * tot["cache_write_passes"])


def hidden_counts(model, n_rows):
    """Phase 2 (gate G1'.1b): the hidden positions of each row, read back from the pad mask that the blocks hold."""
    blocks = [m for m in model.modules() if isinstance(m, LLaDABlock)]
    ms = [m._pad_mask for m in blocks if getattr(m, "_pad_mask", None) is not None]
    if not ms:
        return [0] * n_rows
    assert len(ms) == len(blocks) and all(m is ms[0] for m in ms), "the blocks do not hold one pad mask"
    return (~ms[0]).sum(-1).tolist()


def decode_batch(model, inp, pad_mask, args, sched, ctrl, plog):
    """Phase 2 P5: one batch through the DualCache sampler. `run_cell` and `phase2_padcheck.py` both use it.

    `pad_mask` (rows x (prompt + generation), True = visible) is set on the blocks for this batch, or None. Returns the
    output, the stats, the pass log, the wall time, the time and count of each pass type (CUDA events, no sync in
    the loop), the hidden positions of each row, and the Marlin calls of the batch (None without Marlin).
    """
    if pad_mask is not None:
        set_pad_mask(model, pad_mask)
    hidden = hidden_counts(model, inp.shape[0])
    log, timer = [], []
    if ctrl is not None:
        ctrl.new_block()
    gq = sys.modules.get("gptq_dequant")
    m0 = dict(gq.MarlinLinear.CALLS) if gq is not None else None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out, st = G.generate_with_dual_cache(
            model, inp, steps=args.steps, gen_length=args.gen_length,
            block_length=args.block_length, temperature=0.0, remasking="low_confidence",
            threshold=args.threshold, tau_r=args.tau_r, schedule=sched, controller=ctrl,
            log=log, sink=plog, dus_base=args.dus_base, timer=timer)
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    if pad_mask is not None:
        set_pad_mask(model, None)
    plog.drain()                    # one sync per batch, never inside the pass loop
    t, n = G.pass_times(timer)
    marlin = None if m0 is None else {k: gq.MarlinLinear.CALLS[k] - m0[k] for k in m0}
    return dict(out=out, st=st, log=log, wall=wall, t=t, n=n, hidden=hidden, marlin=marlin)


def run_cell(model, tok, prompts, golds, ids, sched, ctrl, mode, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary_p = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_p):
        print(f">>> SKIP {out_dir}", flush=True)
        return json.load(open(summary_p))
    if ctrl is not None:
        ctrl.mode = mode
    per, steps_f, responses, docs = [], open(os.path.join(out_dir, "steps.jsonl"), "w"), [], []
    B = max(1, int(getattr(args, "batch", 1)))
    plog = PassLog(args.threshold, rows=B > 1)
    eos_t = torch.tensor(args.eos_ids, device=model.device)
    enc = [tok(text).input_ids for text in prompts]
    groups = batch_groups(enc, ids, B)
    done, batch_log = 0, []
    marlin_tot = dict(marlin=0, bf16=0, refine_passes=0, cache_write_passes=0)
    torch.cuda.synchronize()
    for bi, rows in enumerate(groups):
        # Phase 2 P5: a batch is left-padded to its longest prompt and the pads are masked as keys. With B = 1
        # each batch is one problem in the original order, the input is the old one and no mask is set.
        Lp = max(len(enc[i]) for i in rows)
        inp = torch.full((len(rows), Lp), tok.pad_token_id, dtype=torch.long)
        for r, i in enumerate(rows):
            inp[r, Lp - len(enc[i]):] = torch.tensor(enc[i], dtype=torch.long)
        inp = inp.to(model.device)
        padded = any(len(enc[i]) != Lp for i in rows)
        pm = None
        if padded:
            pm = torch.ones((len(rows), Lp + args.gen_length), dtype=torch.bool, device=model.device)
            for r, i in enumerate(rows):
                pm[r, :Lp - len(enc[i])] = False
        d = decode_batch(model, inp, pm, args, sched, ctrl, plog)
        out, st, log, wall = d["out"], d["st"], d["log"], d["wall"]
        if d["marlin"] is not None:
            for k in ("marlin", "bf16"):
                marlin_tot[k] += d["marlin"][k]
            for k in ("refine", "cache_write"):
                marlin_tot[k + "_passes"] += d["n"][k]
        for r, i in enumerate(rows):
            g, pid = golds[i], ids[i]
            # amendment A3: the first end token of the output (-1 if none), measured after the clock
            hit = torch.isin(out[r, Lp:], eos_t).nonzero()
            eos_pos = int(hit[0, 0]) if hit.numel() else -1
            if args.task == "gsm8k":
                gen = tok.decode(out[r, Lp:], skip_special_tokens=True)
                pred = flexible_extract(gen)
                try:
                    ok = pred is not None and abs(float(pred) - float(g)) < 1e-6
                except ValueError:
                    ok = False
            else:                       # P4: scored after the loop, outside the clock
                import phase2_code as PC
                responses.append(PC.response(tok, out[r, Lp:], args.task, args.until))
                docs.append(g)
                pred, ok, g = None, False, None
            rec = dict(problem=int(pid), correct=int(ok), wall_s=wall / len(rows), nfe=int(st),
                       layer_steps=st.layer_steps / len(rows), full_layer_steps=st.full_layer_steps / len(rows),
                       pred=pred, gold=g, eos_pos=eos_pos,
                       # Phase 2 P5: the time of each pass type (a batch's time over its rows), the pads and the
                       # hidden positions of the row (gate G1'.1b)
                       wall_refine_s=d["t"]["refine"] / len(rows), wall_cache_write_s=d["t"]["cache_write"] / len(rows),
                       pads=Lp - len(enc[i]), hidden=int(d["hidden"][r]))
            if B == 1:
                rec["layer_steps"], rec["full_layer_steps"] = int(st.layer_steps), int(st.full_layer_steps)
            else:
                # the row's own passes: the cache-writing pass of each block, and each refinement pass that
                # starts with a masked position of this row in the block
                rec.update(nfe=sum(1 for p in log if p["cache_write"] or p["rows_masked"][r] > 0),
                           nfe_batch=int(st), batch=bi, row=r, wall_batch=wall, prompt_len=len(enc[i]))
            per.append(rec)
        batch_log.append(dict(problems=[int(ids[i]) for i in rows], nfe=int(st), wall_s=wall, padded=padded,
                              prompt_len=Lp, wall_refine_s=d["t"]["refine"], wall_cache_write_s=d["t"]["cache_write"],
                              refine_passes=d["n"]["refine"], cache_write_passes=d["n"]["cache_write"],
                              marlin_calls=d["marlin"]))
        for p in log:
            if B == 1:
                p["problem"] = int(ids[rows[0]])
            else:
                p["batch"] = bi
            steps_f.write(json.dumps(p) + "\n")
        prev, done = done, done + len(rows)
        if done // 50 > prev // 50:
            print(f"    {done}/{len(prompts)}", flush=True)
    steps_f.close()
    if args.task != "gsm8k":
        import phase2_code as PC
        oks, preds = PC.score(CODE["task"], args.task, docs, responses)
        with open(os.path.join(out_dir, "samples.jsonl"), "w") as f:
            for p, ok, r, q in zip(per, oks, responses, preds):
                p["correct"] = int(ok)
                f.write(json.dumps(dict(problem=p["problem"], response=r, program=q[0], passed=int(ok))) + "\n")
    acc = sum(p["correct"] for p in per) / len(per)
    summ = dict(mode=mode, n=len(per), accuracy=acc,
                wall_s=sum(p["wall_s"] for p in per),
                wall_refine_s=sum(b["wall_refine_s"] for b in batch_log),
                wall_cache_write_s=sum(b["wall_cache_write_s"] for b in batch_log),
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
                model_revision=getattr(args, "model_revision", None),
                compression=compression_record(args),
                eos_ids=args.eos_ids, task=args.task, batch_size=B,
                marlin=marlin_record(args, marlin_tot),
                # Phase 2 P5: with B > 1, `nfe` sums the rows' own passes and `nfe_batches` the forward passes
                batches=None if B == 1 else batch_log,
                nfe_batches=sum(b["nfe"] for b in batch_log),
                dus_base=args.dus_base,
                dus_levels=None if args.dus_base is None else G.dus_levels(args.block_length, args.dus_base),
                deterministic=bool(args.deterministic),
                deterministic_strict=bool(args.deterministic_strict), split=args.split,
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
    ap.add_argument("--dus-base", type=int, default=None,
                    help="Phase 1.5: commit by DUS's planned dilated schedule with this base (the "
                         "threshold rule is not used on any pass); cells go under <out>/dus<base>")
    ap.add_argument("--limit", type=int, default=0, help="0 = all of E")
    ap.add_argument("--batch", type=int, default=1,
                    help="Phase 2 P5: problems decoded together. The prompts are sorted by length, cut into batches, "
                         "left-padded, and the pads are masked as keys (model.modeling_llada.set_pad_mask)")
    ap.add_argument("--marlin-row-max", type=int, default=64,
                    help="Phase 2 P5: with --gptq-kernel marlin, calls with more rows than this use the bf16 copy "
                         "(a batch-8 refinement pass has 256 rows)")
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "humaneval", "mbpp"],
                    help="Phase 2 P4: a code task from lm-eval with its default shots (phase2_code.py); --ids is not used")
    ap.add_argument("--gptq", default=None,
                    help="Phase 1b D: a GPTQ snapshot directory; its 4-bit block linears are dequantised into the "
                         "bf16 model after load (gptq_dequant.py). Accuracy / NFE / confidence are the 4-bit "
                         "weights'; the matmul stays bf16, so wall-clock is the bf16 kernel's")
    ap.add_argument("--rtn-bits", type=int, default=None,
                    help="Phase 2: round-to-nearest at load (symmetric, group 128, block linears only; "
                         "gptq_dequant.rtn_quantize)")
    ap.add_argument("--model-revision", default=None,
                    help="Phase 2: the model revision, recorded in summary.json (docs/plan/03_protocol.md section 6)")
    ap.add_argument("--flat", action="store_true",
                    help="Phase 2: write the one cell of this run to <out>/tau<tau_r>/ (protocol section 6), "
                         "not <out>/tau<tau_r>/full/0/ or <out>/tau<tau_r>/static/<mode>/<k>/")
    ap.add_argument("--gptq-kernel", default="dequant", choices=["dequant", "int4", "marlin"],
                    help="dequant: 4-bit weights expanded to bf16 (bf16 matmul); int4: torch's tinygemm int4 kernel "
                         "reads the 4-bit bytes (a real 4-bit wall-clock)")
    ap.add_argument("--gptq-offset", type=int, default=1,
                    help="zero-point offset of the on-disk format (1 = v1; decided by gptq_dequant.py check)")
    ap.add_argument("--deterministic-strict", action="store_true",
                    help="PREREG 0.3 prerequisite 0, closing run: same as --deterministic but "
                         "warn_only=False, so the first operation without a deterministic "
                         "implementation RAISES and is named instead of running with a warning.")
    ap.add_argument("--deterministic", action="store_true",
                    help="PREREG 0.3 prerequisite 0: try to remove the run-to-run floor. Pins the "
                         "SDPA backend to the math kernel, disables cuDNN autotuning and turns on "
                         "torch's deterministic algorithms. CUBLAS_WORKSPACE_CONFIG=:4096:8 must "
                         "be exported by the CALLER -- cuBLAS reads it at init, so setting it "
                         "here would be too late and would silently do nothing.")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    if a.deterministic_strict:
        a.deterministic = True
    if a.deterministic:
        import os as _os
        if _os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise SystemExit("--deterministic needs CUBLAS_WORKSPACE_CONFIG=:4096:8 exported "
                             "before the process starts; cuBLAS reads it at init")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # one kernel, and the one that is deterministic: flash and mem-efficient SDPA both
        # reduce in a nondeterministic order
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=not a.deterministic_strict)
        print(f"[determinism] math SDPA only, cudnn.benchmark off, deterministic algorithms on "
              f"(warn_only={not a.deterministic_strict}), "
              f"CUBLAS_WORKSPACE_CONFIG={_os.environ['CUBLAS_WORKSPACE_CONFIG']}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    # amendment A3: the end tokens of the LLaDA tokenizer, <|endoftext|> and <|eot_id|>
    a.eos_ids = sorted({t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>"))
                        if isinstance(t, int) and t != tok.unk_token_id})
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    if a.gptq:
        import gptq_dequant
        if a.gptq_kernel == "marlin":
            gptq_dequant.load_marlin(model, a.gptq, a.gptq_offset, row_max=a.marlin_row_max)
            a.marlin_linears = sum(isinstance(m, gptq_dequant.MarlinLinear) for m in model.modules())
            print(f"[gptq] block linears replaced by Marlin linears from {a.gptq} (offset {a.gptq_offset}); "
                  f"calls with > {a.marlin_row_max} rows use a bf16 copy of the expanded weights", flush=True)
        elif a.gptq_kernel == "int4":
            gptq_dequant.load_int4pack(model, a.gptq, a.gptq_offset)
            print(f"[gptq] block linears replaced by int4 tinygemm modules from {a.gptq} (offset {a.gptq_offset})",
                  flush=True)
        else:
            gptq_dequant.load_into(model, a.gptq, a.gptq_offset)
            a.gptq_bits = gptq_dequant.checkpoint_bits(a.gptq)
            print(f"[gptq] block linears replaced by dequantised {a.gptq_bits}-bit weights from {a.gptq} "
                  f"(offset {a.gptq_offset})", flush=True)
    # SoloQ hook (docs/plan/03_protocol.md section 5). SoloQ is private: the fork holds only this call. The code
    # is imported from SOLOQ_PATH, a folder outside the fork and outside the project repository. With SOLOQ_PATH
    # set and SOLOQ_WA / SOLOQ_KV unset, patch_from_env must change nothing (the P0b no-op control check).
    if os.environ.get("SOLOQ_PATH"):
        sys.path.insert(0, os.environ["SOLOQ_PATH"])
        import soloq.bridge
        soloq.bridge.patch_from_env(model)
        print(f"[soloq] patch_from_env called; SOLOQ_* = "
              f"{ {k: v for k, v in os.environ.items() if k.startswith('SOLOQ_') and k != 'SOLOQ_PATH'} }", flush=True)
    a.rtn_errors = None
    if a.rtn_bits:
        assert not a.gptq, "--rtn-bits and --gptq exclude each other"
        import gptq_dequant
        a.rtn_errors = gptq_dequant.rtn_quantize(model, a.rtn_bits)
        print(f"[rtn] {a.rtn_bits}-bit round-to-nearest, group 128, block linears; mean relative weight error "
              f"{sum(a.rtn_errors.values()) / len(a.rtn_errors):.5f}", flush=True)
    ids_obj = json.load(open(a.ids)) if a.task == "gsm8k" else dict(E=[], split="test")
    E = ids_obj["E"]
    # results/README rule 4: the split is a registered input, so it is asserted and logged rather
    # than inferred. Reading test-split indices against the train split would silently score the
    # wrong 1319 problems and every number downstream would be wrong but plausible.
    # The ids file's `split` field is not guaranteed to BE a split name: phase0.25/ids.json
    # records the human label "gsm8k train". Treating it as one built the pattern
    # `gsm8k train-*.parquet` and died -- loudly, which is lucky, because a label of "train"
    # would have worked by accident. So it is normalised (last whitespace token) and then
    # validated; anything that does not resolve to train/test is reported as a label and
    # --split decides.
    raw = str(ids_obj.get("split", "") or "")
    file_split = raw.split()[-1].lower() if raw.split() else ""
    if file_split not in ("train", "test"):
        if raw:
            print(f"note: {a.ids} declares split={raw!r}, which is a label, not a split name; "
                  f"using --split", flush=True)
        file_split = ""
    split = a.split or file_split or "train"
    if a.split and file_split and a.split != file_split:
        raise SystemExit(f"--split {a.split} but {a.ids} resolves to split={file_split} "
                         f"(from {raw!r})")
    if split not in ("train", "test"):
        raise SystemExit(f"split must be train or test, got {split!r}")
    a.split = split
    if a.limit:
        E = E[:a.limit]
    if a.task == "gsm8k":
        prompts, golds = build(tok, E, a.n_shot, split)
    else:
        import phase2_code as PC
        CODE["task"], prompts, golds, E, a.until = PC.build(tok, a.task, a.limit)
        print(f"task {a.task}: {len(E)} problems from lm-eval, stop sequences {a.until}", flush=True)
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
    src = f"GSM8K-{split} ({a.ids})" if a.task == "gsm8k" else f"lm-eval {a.task}"
    print(f"E: {len(E)} problems from {src} | L={L} | regimes: {regimes} | "
          f"r* = {a.r_star} | tau_w = {a.threshold} | tau_r = {a.tau_r} | cells: {a.cells}",
          flush=True)
    for k, (st, rules, label, path) in sorted(kl_sets.items()):
        print(f"  KL set k={k:<3} {st}  rules={rules}  label={label!r}  <- {path}", flush=True)

    # PREREG 0.3 sweeps tau_r, so the cell path carries it; without this a sweep would write
    # every tau into the same directory and resume would skip the rest of it.
    if a.tau_r is not None:
        a.out = os.path.join(a.out, f"tau{a.tau_r:g}")
    if a.batch > 1:
        assert a.dus_base is None and regimes == ["static"], \
            "--batch > 1 needs the threshold rule and the static regime (the mask ratio of a pass is a batch mean)"
    if a.flat:
        # protocol section 6: one folder per cell, so a flat run holds exactly one cell
        assert len([c for c in a.cells.split(",") if c.strip()]) == 1 and regimes == ["static"], \
            "--flat needs one cell and the static regime"
    if a.dus_base is not None:
        assert a.tau_r is None, "a DUS cell reads no threshold; --tau-r would be silently ignored"
        a.out = os.path.join(a.out, f"dus{a.dus_base}")
        print(f"DUS base {a.dus_base}: levels per block {G.dus_levels(a.block_length, a.dus_base)}", flush=True)
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
                     a.out if a.flat else os.path.join(a.out, "full", "0"))
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
                         a.out if a.flat else os.path.join(a.out, regime, "identity-kl", str(k)))
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
                     a.out if a.flat else os.path.join(a.out, regime, name, str(k)))
    uninstall_skipping(model)


if __name__ == "__main__":
    main()
