"""Phase 0.25 Stage 2 runner, Family B — `results/phase0.25/PREREG.md` §2-§5.

Family A's runner with the gate setting of `06` §2 step 5(c): Fast-dLLM v2 7B, tau 0.9,
batch 1, block 32, `small_block_size = 32`, `use_block_cache = False`, **gen 512** (256
truncated 62 % of generations, `../phase0/fix512_L100/SUMMARY.md`), 0-shot chat template with
the boxed-answer instruction `v2/eval.py` builds.

Confidence is the loop's own `x1_p`, handed to the sink by the instrumented `batch_sample`.

Phase 3 (results/phase3/PREREG.md; amendments A13, changes 6, 7 and 11; A14, changes 5, 12, 13 and 14; A15,
changes 3 and 4). The Phase 3 jobs pass --check-a7 (block 32, small block 8, 2,048 new tokens, the test problems, the
model revision, the static regime), and also --small-block-size 8 --max-new-tokens 2048 (A7).
- The NFE counts the refinement passes and the cache-writing passes of the generation, not the one pass over the prompt
  (A13, change 6; A15, change 3). Each problem records the count and the time of each pass type (prompt, refine,
  cache_write; CUDA events, no sync in the loop) and the calls of each path of the 4-bit kernels for each pass type.
  `nfe_with_prompt` keeps the old count (gate M1 and Phases 0.3 and 2).
- The stop (the sampler does not change): the block and the pass at which the sampler stops, the block of the first
  end token, the pass that commits it, and the passes after it. The summary gives the number of outputs that go on
  after the block of their first end token (A15, change 3).
- GSM8K: `correct` is the scorer of gate M1 (the last boxed answer, else the last number, of the whole decoded output).
  Also recorded with no reading: the same scorer on the text cut at the first end token, and lm-eval
  "flexible-extract" on both texts (A15, change 5: C2 of SoloQ).
- Code tasks: the position of the closing code fence and the NFE of the blocks up to it (A13, change 11; A14,
  change 6).
- Each summary.json records the fork commit and `git status`, the job, the restart, the host, the GPU and its UUID,
  the start and end time of the cell, the task (data set, revision, prompt style, input checks), the shard, the bar
  (one field), the SoloQ state (config_from_env(); a cell that is not a SoloQ cell stops if SoloQ gives a
  configuration) and the sha256 of the compressed weights. A finished cell is skipped only if it has the same
  arguments and fork commit (and, with --one-allocation, the same job and restart).
- --warmup decodes the first problem of the shard once before the timed problems (A13, change 7). --trace-problems
  traces one pass of each type in --trace-kinds with torch.profiler (A14, change 13).
- identity-kl reads the set and its pool rules from --kl-json (no default) and checks the rules against A4, change 3
  (L = 28: keep layer 0 and the last 7, no two adjacent, k = 6). Each refinement pass must skip exactly the set
  (A14, change 14, rule 7).
"""
import argparse, json, os, re, sys, time, types
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import generation_functions                                             # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "v1", "llada")))
import phase3_tasks as P3                                               # noqa: E402
import phase3_records as R                                              # noqa: E402
import profile_step_v2 as P                                             # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM            # noqa: E402
from dllm_skip.hook import install_skipping, uninstall_skipping, MODES  # noqa: E402
from dllm_skip.depth_schedule import (DepthSchedule, RegimeSchedule, StepState,   # noqa: E402
                                      leq_share)

MASK_ID = 151665
STOP_TOKEN = 151645                 # <|im_end|>: the sampler stops when a refinement pass commits it
LAYERS = 28
ANS = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
BOXED = re.compile(r"boxed\{([^}]*)\}")
A7 = dict(block_size=32, small_block_size=8, max_new_tokens=2048)


def extract(text):
    m = BOXED.findall(text)
    src = m[-1] if m else text
    g = ANS.findall(src)
    if not g:
        return None
    last = [t for t in g[-1] if t][-1]
    return last.replace("$", "").replace(",", "").rstrip(".")


def num_eq(pred, gold):
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


class PassLog:
    """The statistics of each pass, reduced on the GPU and moved to the host once per problem. `thr` is the bar of the
    cell (n_ge_tau). With `eos_ids`, `commits` keeps the end tokens that each pass commits (batch 1): the record gets
    `eos_commit` (the absolute positions of the end tokens it commits) and `stop_commit` (those of <|im_end|>)."""

    def __init__(self, thr, eos_ids=None, device=None):
        self._pending, self.thr = [], thr
        self._commits = []
        self._eos = None if eos_ids is None else torch.tensor(eos_ids, device=device)

    def add(self, rec, conf, mask, committed):
        c = conf[mask]
        self._pending.append((rec, torch.stack(
            [c.mean(), c.min(), (c >= self.thr).sum().to(c.dtype),
             committed.sum().to(c.dtype), (c >= 0.9).sum().to(c.dtype)]) if c.numel() else None))

    def commits(self, rec, tokens, committed, pos0):
        if self._eos is None:
            return
        code = (torch.isin(tokens[0], self._eos) & committed[0]).to(torch.int8) \
            + ((tokens[0] == STOP_TOKEN) & committed[0]).to(torch.int8)
        self._commits.append((rec, code, pos0))

    def drain(self):
        vals = [None if t is None else t.tolist() for _, t in self._pending]
        for (rec, _), v in zip(self._pending, vals):
            if v is not None:
                rec["conf_mean"], rec["conf_min"], rec["n_ge_tau"], rec["committed"] = v[:4]
                rec["n_ge_09"] = v[4]           # amendment A6: the positions >= 0.9 on every pass
        self._pending = []
        if self._commits:
            flat = torch.cat([c for _, c, _ in self._commits]).tolist()     # one sync
            i = 0
            for rec, c, pos0 in self._commits:
                w = c.numel()
                part = flat[i:i + w]
                i += w
                e = [pos0 + j for j, v in enumerate(part) if v >= 1]
                if e:
                    rec["eos_commit"] = e
                    st = [pos0 + j for j, v in enumerate(part) if v == 2]
                    if st:
                        rec["stop_commit"] = st
            self._commits = []


def compression_record(a):
    """docs/plan/03_protocol.md section 6: kind, bits, method, revision. The SoloQ state is in `soloq` of the
    summary (phase3_records.soloq_state)."""
    if getattr(a, "compress_rec", None):
        return dict(a.compress_rec)
    if getattr(a, "soloq", False):
        return dict(kind="weights+activations", bits=None, method="soloq", revision=None)
    return dict(kind="none", bits=None, method=None, revision=None)


def stop_record(log, L, eos_pos, block_size):
    """The stop of one problem (A15, change 3; records only). `log`: the pass records, in order."""
    gen = [r for r in log if r.get("kind") != "prompt"]
    last_block = max((r["block"] for r in gen), default=None)
    rec = dict(stop_block=last_block, stop_pass=len(gen), eos_block=None, eos_commit_pass=None,
               passes_after_eos=None, passes_after_eos_block=None, blocks_after_eos=None,
               stop_token_committed=any("stop_commit" in r for r in gen))
    if eos_pos >= 0:
        pos = L + eos_pos
        rec["eos_block"] = eos_block = pos // block_size
        rec["passes_after_eos_block"] = sum(1 for r in gen if r["block"] > eos_block)
        rec["blocks_after_eos"] = None if last_block is None else last_block - eos_block
        for i, r in enumerate(log):
            if pos in r.get("eos_commit", ()):
                n_before = sum(1 for q in log[:i + 1] if q.get("kind") != "prompt")
                rec["eos_commit_pass"] = n_before            # 0 if the pass over the prompt commits it
                rec["passes_after_eos"] = len(gen) - n_before
                break
    return rec


def write_fsync(path, lines):
    """Phase 3 amendment A14, change 12: the record files are on the disk before summary.json."""
    with open(path + ".tmp", "w") as f:
        for line in lines:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


def decode(model, tok, text, sched, ctrl, a, plog, timer):
    ids_t = tok([text], return_tensors="pt").input_ids.to(model.device)
    L = ids_t.shape[1]
    log = []
    if ctrl is not None:
        ctrl.new_block()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model.mdm_sample(
            ids_t, tokenizer=tok, block_size=a.block_size,
            small_block_size=a.small_block_size, max_new_tokens=a.max_new_tokens,
            mask_id=MASK_ID, min_len=L, seq_len=torch.tensor([L], device=model.device),
            use_block_cache=False, threshold=a.threshold, tau_r=a.tau_r,
            schedule=sched, controller=ctrl, log=log, sink=plog, timer=timer)
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    plog.drain()                        # one sync per problem, never inside the pass loop
    return out[0], L, log, wall         # mdm_sample: {index: 1-D ids}


def run_cell(model, tok, prompts, golds, ids, sched, ctrl, mode, a, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sp = os.path.join(out_dir, "summary.json")
    if os.path.exists(sp):
        old = json.load(open(sp))
        # Phase 3 amendment A14, change 12: a finished cell is this cell only if its arguments and fork commit are
        # those of this run (and, with --one-allocation, its job and restart)
        bad = R.same_args(old, dict(args=vars(a), run=a.run), a.one_allocation)
        if bad:
            raise SystemExit(f"{out_dir} holds a cell of another run: {'; '.join(bad)}")
        print(f">>> SKIP {out_dir}", flush=True); return old
    if ctrl is not None:
        ctrl.mode = mode
    t_start = R.now()
    print(f">>> START {out_dir} {t_start}", flush=True)
    per, steps, samples = [], [], []    # samples: one record per problem (output, solution, answer)
    eos_t = torch.tensor(a.eos_ids, device=model.device)
    path_tot = {}
    if a.warmup:
        # Phase 3 amendment A13, change 7: the first problem of the shard once, not timed and not saved
        decode(model, tok, prompts[0], sched, ctrl, a, PassLog(a.bar), None)
    traced = set()
    if a.trace_problems:
        traced = {int(x) % len(prompts) for x in a.trace_problems.split(",")}
    for n, (text, g, pid) in enumerate(zip(prompts, golds, ids)):
        timer = R.PassTracer(a.trace_pass, os.path.join(out_dir, f"trace_problem{int(pid)}.json"),
                             a.trace_kinds.split(",")) if n in traced else R.PassTimer()
        plog = PassLog(a.bar, a.eos_ids, model.device)
        out, L, log, wall = decode(model, tok, text, sched, ctrl, a, plog, timer)
        t, cnt, calls = R.pass_stats(timer)
        gen_ids = out[L:]
        # amendment A3: the first end token of the output (-1 if none), measured after the clock
        hit = torch.isin(gen_ids, eos_t).nonzero()
        eos_pos = int(hit[0, 0]) if hit.numel() else -1
        extra = {}
        if a.task == "gsm8k":
            gen = tok.decode(gen_ids, skip_special_tokens=True)
            cut = tok.decode(gen_ids if eos_pos < 0 else gen_ids[:eos_pos], skip_special_tokens=True)
            pred, pred_cut = extract(gen), extract(cut)
            ok, ok_cut = num_eq(pred, g), num_eq(pred_cut, g)
            flex, _, flex_pred = P3.gsm8k_scores(gen, g)
            flex_cut, _, flex_pred_cut = P3.gsm8k_scores(cut, g)
            # Phase 3 amendment A10, change 7: gate M1 scores the outputs with a second scorer, so keep them
            samples.append(dict(problem=int(pid), response=gen, pred=pred, gold=g, correct=int(ok),
                                response_cut=cut, pred_cut=pred_cut, correct_cut=int(ok_cut), flex_pred=flex_pred,
                                correct_flex=int(flex), flex_pred_cut=flex_pred_cut, correct_flex_cut=int(flex_cut),
                                eos_pos=eos_pos))
            extra = dict(correct_cut=int(ok_cut), correct_flex=int(flex), correct_flex_cut=int(flex_cut))
        else:
            # Phase 3 (amendment A6): the task module of v1/llada; MATH-500 is scored here, a code task later
            resp = P3.response(tok, gen_ids, eos_pos, a.until)
            samples.append(P3.sample_record(a.task, pid, g, resp))
            if a.task == "math500":
                ok, pred = P3.math_correct(resp, g["answer"])
                samples[-1].update(pred=pred, gold=g["answer"], correct=int(ok))
            else:
                ok, pred = None, None
            if a.task in P3.CODE_TASKS:
                # A13, change 11 and A14, change 6: the closing code fence and the NFE of the blocks up to it
                fc, ft = P3.fence_position(tok, gen_ids.tolist(), eos_pos)
                fb = None if ft is None else (L + ft) // a.block_size
                nfe_fence = None if fb is None else sum(1 for r in log if r["kind"] != "prompt" and r["block"] <= fb)
                samples[-1].update(fence_char=fc, fence_tok=ft)
                extra = dict(fence_tok=ft, fence_block=fb, nfe_to_fence=nfe_fence)
            g = g["answer"] if a.task == "math500" else None
        kinds = [r["kind"] for r in log]
        n_prompt, n_refine, n_cw = kinds.count("prompt"), kinds.count("refine"), kinds.count("cache_write")
        assert (cnt.get("prompt", 0), cnt["refine"], cnt["cache_write"]) == (n_prompt, n_refine, n_cw), \
            f"the timer counts {cnt} are not the pass records ({n_prompt}, {n_refine}, {n_cw})"
        gen_log = [r for r in log if r["kind"] != "prompt"]
        rec = dict(problem=int(pid), correct=None if ok is None else int(ok), wall_s=wall,
                   nfe=n_refine + n_cw, nfe_with_prompt=len(log), n_prompt=n_prompt, n_refine=n_refine,
                   n_cache_write=n_cw, wall_prompt_s=t.get("prompt", 0.0), wall_refine_s=t["refine"],
                   wall_cache_write_s=t["cache_write"], pred=pred, gold=g,
                   layer_steps=sum(r["depth"] for r in gen_log), full_layer_steps=len(gen_log) * LAYERS,
                   eos_pos=eos_pos, prompt_len=L,
                   # Phase 3 (amendment A7): a cut output reaches the limit before its end token
                   n_new=int(out.shape[0] - L), cut=int(eos_pos < 0), **extra)
        rec.update(stop_record(log, L, eos_pos, a.block_size))
        if any(v for c in calls.values() for v in c.values()):
            rec["path_calls"] = calls
            R.add_calls(path_tot, calls)
        if n in traced:
            rec["traced"] = timer.done              # the trace adds a sync: this time is not clean
        per.append(rec)
        for r in log:
            r["problem"] = int(pid)
            steps.append(json.dumps(r))
        if (n + 1) % 50 == 0:
            print(f"    {n+1}/{len(prompts)}", flush=True)
    write_fsync(os.path.join(out_dir, "steps.jsonl"), steps)
    write_fsync(os.path.join(out_dir, "samples.jsonl"), [json.dumps(r) for r in samples])
    import glob
    traces = {}
    for i in sorted(traced):
        for f in sorted(glob.glob(os.path.join(out_dir, f"trace_problem{int(ids[i])}*.json"))):
            traces[os.path.basename(f)] = R.gpu_share(f)
    scored = all(p["correct"] is not None for p in per)
    nl = ctrl.n_layers if ctrl is not None else LAYERS
    summ = dict(mode=mode, n=len(per), accuracy=sum(p["correct"] for p in per) / len(per) if scored else None,
                scoring="done" if scored else "pending", task=a.task, task_record=P3.task_record(a.task),
                shard=a.shard_info, cut_share=sum(p["cut"] for p in per) / len(per),
                settings=dict(block_size=a.block_size, small_block_size=a.small_block_size,
                              max_new_tokens=a.max_new_tokens, use_block_cache=False, check_a7=a.check_a7),
                bar=a.bar, wall_s=sum(p["wall_s"] for p in per), nfe=sum(p["nfe"] for p in per),
                nfe_with_prompt=sum(p["nfe_with_prompt"] for p in per),
                passes={k: sum(p[f"n_{k}"] for p in per) for k in ("prompt", "refine", "cache_write")},
                wall_pass_s={k: sum(p[f"wall_{k}_s"] for p in per) for k in ("prompt", "refine", "cache_write")},
                layer_steps=sum(p["layer_steps"] for p in per),
                full_layer_steps=sum(p["nfe"] for p in per) * nl,
                # A15, change 3: the outputs that go on after the block of their first end token
                n_after_eos_block=sum(1 for p in per if (p["blocks_after_eos"] or 0) > 0),
                budget=None if sched is None else sched.budget,
                # PREREG 0.4 §2: the regime and its boundary travel with the cell. A regime
                # cell and the static cell it is compared with differ in exactly one field, and
                # this is it -- reading them apart from the directory name would be a guess.
                regime=getattr(sched, "regime", "static"),
                r_star=getattr(sched, "r_star", None),
                tau_w=a.threshold, tau_r=a.tau_r,
                model_revision=a.model_revision, model_snapshot=a.model_snapshot,
                compression=compression_record(a), eos_ids=a.eos_ids, split=a.split,
                # Phase 3 amendment A10, change 8: a paired comparison uses one GPU type
                gpu=torch.cuda.get_device_name(0),
                # Phase 3 amendment A13, change 11 and A14, change 12: the records of the cell
                run=dict(a.run, cell_start=t_start, cell_end=R.now()), soloq=a.soloq_rec,
                weights_sha256=a.weights_sha256, warmup=a.warmup, kl_set=getattr(sched, "kl_record", None),
                path_calls=path_tot, traces=traces or None, args=vars(a), per_problem=per)
    if a.task == "gsm8k":
        for k in ("correct_cut", "correct_flex", "correct_flex_cut"):
            summ["accuracy_" + k.split("_", 1)[1]] = sum(p[k] for p in per) / len(per)
    # Raw layer count, then the byte-weighted L_eq ratio the gate axis is defined in:
    # a skipped `no-attn` layer still runs its FFN, so counting it as zero understates the
    # cost badly (0.10 instead of ~0.87 in Family B).
    fls = summ["full_layer_steps"]
    summ["depth_ratio_layers"] = summ["layer_steps"] / fls if fls else float("nan")
    share = leq_share(mode, LAYERS)
    skipped_layer_steps = fls - summ["layer_steps"]
    summ["leq_removed"] = skipped_layer_steps * share
    summ["depth_ratio"] = 1.0 - (summ["leq_removed"] / fls) if fls else float("nan")
    # summary.json marks the cell as done, so write it whole or not at all: a job stopped at any moment must
    # not leave a cut file that a rerun skips as done (the user's rule of 2026-10-24)
    with open(sp + ".tmp", "w") as f:
        json.dump(summ, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(sp + ".tmp", sp)
    acc_s = "pending" if summ["accuracy"] is None else f"{summ['accuracy']:.4f}"
    print(f">>> DONE {out_dir}  acc={acc_s}  cut={summ['cut_share']:.3f}  wall={summ['wall_s']:.1f}s  "
          f"nfe={summ['nfe']}  depth={summ['depth_ratio']:.4f}  after-eos-block={summ['n_after_eos_block']}  "
          f"end {summ['run']['cell_end']}", flush=True)
    return summ


def kl_set(path, L):
    """The layer set of identity-kl and its pool rules, from the set file; the rules must be those of A4, change 3
    (A14, change 14, rule 7)."""
    j = json.load(open(path))
    klset = sorted(int(x) for x in j["kl_greedy_set"])
    rules = j.get("rules")
    if rules is None:
        raise SystemExit(f"{path} has no pool rules")
    want = dict(keep_first=1, keep_last=round(L / 4), no_consecutive=True)
    got = dict(keep_first=int(rules["keep_first"]), keep_last=int(rules["keep_last"]),
               no_consecutive=bool(rules["no_consecutive"]))
    if got != want or len(klset) != round(0.22 * L):
        raise SystemExit(f"{path}: rules {got} and k {len(klset)}, not those of A4, change 3 ({want}, "
                         f"k {round(0.22 * L)})")
    return klset, got, dict(file=path, set=klset, rules=got, k=len(klset))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    ap.add_argument("--ids", default="/home1/hyunhoko/DLLM/results/phase0.25/ids.json")
    ap.add_argument("--calib", default="",
                    help="calibrate_depth_v2.py output; supplies the layer ranking")
    ap.add_argument("--kl-json", default=None,
                    help="the set file of identity-kl, with its pool rules (no default: A14, change 14)")
    ap.add_argument("--regimes", default="static",
                    help="comma-separated: early | late | static (PREREG 0.4 §2). `early` skips "
                         "only where the block mask ratio r > r*, `late` only where r <= r*.")
    ap.add_argument("--split", default=None, choices=["train", "test"],
                    help="which GSM8K split the ids index; read from the ids file when omitted, "
                         "and the two must agree")
    ap.add_argument("--r-star", type=float, default=None,
                    help="the regime boundary, measured in Phase 0.35. Required unless every "
                         "regime is `static`.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--small-block-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--tau-r", type=float, default=None,
                    help="the threshold on refinement passes (PREREG 0.3 §2). Family B commits "
                         "only there -- its prefill and clean-block encode take the argmax -- so "
                         "there is no separate tau_w. A regime row applies it only on the passes "
                         "its regime actually skips.")
    ap.add_argument("--keep-last", type=int, default=8,
                    help="the cosine-order cells of Phases 0.25 to 0.4 only; identity-kl takes its rules from the set "
                         "file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None,
                    help="Phase 3 amendment A12: I/K, the problems at the positions j with j mod K = I; --out must "
                         "name the shard (shardIofK)")
    ap.add_argument("--compress", default=None,
                    choices=["fp8", "awq4", "quarot_w4a4", "sparse24", "sparse24_masked", "gptq4"],
                    help="Phase 3: a compression of v1/llada/phase3_compress.py, put into the model after load")
    ap.add_argument("--compress-dir", default=None, help="Phase 3: the artifact folder of this model")
    ap.add_argument("--kernel-row-max", type=int, default=64,
                    help="Phase 3: the 4-bit kernels (awq4, gptq4) run on calls with at most this many rows")
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k"] + [t for t in P3.TASKS if t != "gsm8k_lmeval"],
                    help="Phase 3: a task of v1/llada/phase3_tasks.py (prompts of results/phase3/references.md, "
                         "section 1, with the exact prompts of A14, change 5); gsm8k keeps the prompt of the official "
                         "code")
    ap.add_argument("--model-revision", default=None,
                    help="the model revision: it goes to from_pretrained and to summary.json (A14, change 5)")
    ap.add_argument("--flat", action="store_true",
                    help="Phase 2: write the one cell of this run to <out>/tau<tau_r>/ (protocol section 6)")
    ap.add_argument("--check-a7", action="store_true",
                    help="Phase 3 amendment A15, change 3: stop unless the settings are those of A7 (block 32, small "
                         "block 8, 2,048 new tokens, the test problems), the model revision is given and the regime "
                         "is static")
    ap.add_argument("--warmup", action="store_true",
                    help="Phase 3 amendment A13, change 7: decode the first problem once before the timed problems")
    ap.add_argument("--one-allocation", action="store_true",
                    help="Phase 3 amendment A14, change 12: a finished cell of another job or restart stops the run")
    ap.add_argument("--soloq", action="store_true",
                    help="Phase 3 amendment A15, change 4: this is a SoloQ cell (SOLOQ_PATH set, a configuration "
                         "active). Without it, an active SoloQ configuration stops the run")
    ap.add_argument("--trace-problems", default=None,
                    help="Phase 3 (A14, change 13): positions of problems in the cell (for example 0,-1) whose pass "
                         "--trace-pass of each type in --trace-kinds is traced with torch.profiler")
    ap.add_argument("--trace-pass", type=int, default=2, help="the pass of each traced type to trace (1 = the first)")
    ap.add_argument("--trace-kinds", default="refine,cache_write")
    a = ap.parse_args()

    dev = torch.device("cuda")
    a.run = R.run_record()
    print(f"run: {a.run}", flush=True)
    regimes = [r.strip() for r in a.regimes.split(",") if r.strip()]
    a.bar = a.tau_r if a.tau_r is not None else a.threshold     # A15, change 3: the bar of the cell, one field
    if a.check_a7:
        got = dict(block_size=a.block_size, small_block_size=a.small_block_size, max_new_tokens=a.max_new_tokens)
        if got != A7 or a.model_revision is None or regimes != ["static"]:
            raise SystemExit(f"--check-a7: settings {got} (A7: {A7}), revision {a.model_revision}, regimes {regimes}")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.model_revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.model_revision, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    a.model_snapshot = R.snapshot(a.model, a.model_revision)
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model)
    # amendment A3: the end tokens of the Qwen tokenizer
    a.eos_ids = sorted({t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                                    tok.convert_tokens_to_ids("<|endoftext|>"))
                        if isinstance(t, int) and t != tok.unk_token_id})
    # SoloQ hook (docs/plan/03_protocol.md section 5). SoloQ is private: the fork holds only this call. The code is
    # imported from SOLOQ_PATH, a folder outside the fork and the project repository. Phase 3 amendment A15, change 4:
    # SOLOQ_PATH is set only for a SoloQ cell (--soloq); SoloQ's own config_from_env() decides if a configuration is
    # active, and a cell that is not a SoloQ cell stops if one is (phase3_records.soloq_state).
    a.soloq_rec = R.soloq_state(model, a.soloq)
    print(f"[soloq] {a.soloq_rec}", flush=True)
    a.compress_rec = None
    if a.compress:
        assert not a.soloq, "one compression per cell"
        import phase3_compress as PCX
        a.compress_rec = PCX.apply(model, a.compress, a.compress_dir, row_max=a.kernel_row_max, model_name=a.model,
                                   revision=a.model_revision)
        print(f"[compress] {a.compress}: {a.compress_rec.get('kernel')} ({a.compress_rec.get('artifact')})", flush=True)
    # Phase 3 amendment A13, change 11: the sha256 of each file of compressed weights
    a.weights_sha256 = None
    if a.compress_rec and a.compress_rec.get("artifact"):
        a.weights_sha256 = {a.compress_rec["artifact"]: R.sha256(a.compress_rec["artifact"])}
        print(f"[weights] sha256 {a.weights_sha256}", flush=True)

    ids_obj = json.load(open(a.ids))
    E = ids_obj["E"]
    if a.limit:
        E = E[:a.limit]
    # results/README rule 4. The split was inferred from the ids FILENAME, which is a guess that
    # scores a different 1319 problems when it is wrong. It is now read from the file's own
    # `split` field (normalised -- phase0.25/ids.json records the label "gsm8k train"),
    # cross-checked against --split, validated, and logged.
    raw = str(ids_obj.get("split", "") or "")
    file_split = raw.split()[-1].lower() if raw.split() else ""
    if file_split not in ("train", "test"):
        if raw:
            print(f"note: {a.ids} declares split={raw!r}, a label rather than a split name; "
                  f"using --split", flush=True)
        file_split = ""
    split = a.split or file_split or "train"
    if a.split and file_split and a.split != file_split:
        raise SystemExit(f"--split {a.split} but {a.ids} resolves to split={file_split} "
                         f"(from {raw!r})")
    if split not in ("train", "test"):
        raise SystemExit(f"split must be train or test, got {split!r}")
    a.split = split
    if a.check_a7 and a.task == "gsm8k" and split != "test":
        raise SystemExit("--check-a7: the GSM8K cells use the test problems")
    prompts, golds = [], []
    a.until = []
    P3.EXACT_PROMPTS = True             # A14, change 5: the exact prompts of the models after LLaDA-8B-Instruct
    if a.task != "gsm8k":
        prompts, golds, E, a.until = P3.build(tok, a.task, a.limit)
        print(f"task {a.task}: {len(E)} problems, stop strings {a.until}", flush=True)
    else:
        q, ans = P._gsm8k(split)
        for i in E:
            text = f"Question: {q[i]}\nAnswer:".replace("Answer:", P.GSM8K_INSTRUCTION)
            prompts.append(tok.apply_chat_template([{"role": "user", "content": text}],
                                                   add_generation_prompt=True, tokenize=False))
            golds.append(ans[i].split("####")[-1].strip().replace(",", ""))
    a.shard_info = None
    if a.shard:
        # Phase 3 amendment A12: the problem at position j is in shard j mod K; the records keep the problem ids
        si, sk = (int(x) for x in a.shard.split("/"))
        keep = [j for j in range(len(E)) if j % sk == si]
        prompts, golds, E = [prompts[j] for j in keep], [golds[j] for j in keep], [E[j] for j in keep]
        a.shard_info = dict(shard=si, of=sk, n=len(E))
        print(f"shard {si} of {sk}: {len(E)} problems (positions j with j mod {sk} = {si})", flush=True)
        # Phase 3 amendment A14, change 12: stop on an empty shard; the shard is in the path of the cell
        if not E:
            raise SystemExit(f"shard {si} of {sk} has no problem")
        if f"shard{si}of{sk}" not in os.path.normpath(a.out).split(os.sep):
            raise SystemExit(f"--out {a.out} does not name the shard (shard{si}of{sk})")
    print(f"task record: {P3.task_record(a.task)}", flush=True)

    ctrl = install_skipping(model)
    L = ctrl.n_layers
    # The layer ranking skip sets are drawn from. With `--calib` this is Family B's own
    # adjacent-layer cosine (`calibrate_depth_v2.py`), which is what `PREREG.md` §3 registers.
    # Without it the fallback is the natural layer order -- and under non-adjacency that is
    # exactly the naive {1, 3, 5, ...} set §3 excludes by name, which is how the first Family B
    # grid came to run on the excluded sets (`RESULTS.md`, Deviations 1). Left in place so that
    # grid stays reproducible; every new run passes --calib.
    if a.calib:
        cal = json.load(open(a.calib))
        assert cal["n_layers"] == L, f"calibration has L={cal['n_layers']}, model has {L}"
        order = list(cal["global_order"])
        print(f"skip order from {a.calib}: {order[:12]}", flush=True)
    else:
        order = list(range(1, L - 1))
        print("skip order: natural (NO CALIBRATION -- see RESULTS.md Deviations 1)", flush=True)
    # results/README rule 4: assert and log every registered input; no silent fallback.
    for r in regimes:
        assert r in ("early", "late", "static"), f"unknown regime {r!r}"
    if any(r != "static" for r in regimes):
        assert a.r_star is not None, "--r-star is required for a non-static regime (Phase 0.35)"
    print(f"E: {len(E)} problems from GSM8K-{split} ({a.ids}) | L={L} | regimes: {regimes} | "
          f"r* = {a.r_star} | tau_r = {a.tau_r} | bar = {a.bar} | cells: {a.cells}", flush=True)
    if a.tau_r is not None:
        a.out = os.path.join(a.out, f"tau{a.tau_r:g}")
    if a.flat:
        assert len([c for c in a.cells.split(",") if c.strip()]) == 1 and regimes == ["static"], \
            "--flat needs one cell and the static regime"
    for spec in a.cells.split(","):
        spec = spec.strip()
        if spec == "full":
            sch = DepthSchedule.static(L, [order], 0, keep_first=1, keep_last=a.keep_last)
            sch.expect_skip = set()                 # A14, change 14, rule 7: each refinement pass is full depth
            # a full-depth run has no regime, so it is written once, outside the regime tree
            run_cell(model, tok, prompts, golds, E, sch, ctrl, "identity", a,
                     a.out if a.flat else os.path.join(a.out, "full", "0")); continue
        mode, k = spec.split(":")
        # `identity-kl` takes its skip set from the KL-greedy search on S (`PREREG.md` §3)
        # instead of the cosine ranking -- the one cell that asks whether selection quality,
        # not budget size, moves the pass count.
        if mode == "identity-kl":
            if not a.kl_json:
                raise SystemExit("identity-kl needs --kl-json (the set file; no default)")
            klset, rules, kl_rec = kl_set(a.kl_json, L)
            assert len(klset) == int(k), f"kl set has {len(klset)} layers, cell asks {k}"
            kl_order = klset + [l for l in range(L) if l not in klset]
            sch = DepthSchedule.static(L, [kl_order], int(k), keep_first=rules["keep_first"],
                                       keep_last=rules["keep_last"], no_consecutive=rules["no_consecutive"])
            got = sorted(set(range(L)) - set(sch.active_layers(StepState(0.5))))
            assert got == sorted(klset), f"schedule picked {got}, not the KL set {sorted(klset)}"
            sch.expect_skip = set(klset)            # A14, change 14, rule 7: checked on each refinement pass
            sch.kl_record = kl_rec
            print(f"KL set k={k} {klset} rules={rules} <- {a.kl_json}", flush=True)
            ctrl.reuse_m = None
            for regime in regimes:
                sc = sch if regime == "static" else \
                    RegimeSchedule(base=sch, regime=regime, r_star=a.r_star)
                run_cell(model, tok, prompts, golds, E, sc, ctrl, "identity", a,
                         a.out if a.flat else os.path.join(a.out, regime, "identity-kl", str(k)))
            continue
        # `reuse2` / `reuse4` / `reuse` select the refresh interval m (reuse = inf)
        m = None
        if mode.startswith("reuse") and mode != "reuse":
            m = int(mode[len("reuse"):]); mode = "reuse"
        ctrl.reuse_m = m
        assert mode in MODES, mode
        nc = mode in ("identity", "reuse")
        sch = DepthSchedule.static(L, [order], k, keep_first=1,
                                   keep_last=a.keep_last if nc else 0, no_consecutive=nc)
        name = mode if m is None else f"{mode}{m}"
        for regime in regimes:
            sc = sch if regime == "static" else \
                RegimeSchedule(base=sch, regime=regime, r_star=a.r_star)
            run_cell(model, tok, prompts, golds, E, sc, ctrl, mode, a,
                     os.path.join(a.out, regime, name, str(k)))
    uninstall_skipping(model)


if __name__ == "__main__":
    main()
