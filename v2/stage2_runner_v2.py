"""Phase 0.25 Stage 2 runner, Family B — `results/phase0.25/PREREG.md` §2-§5.

Family A's runner with the gate setting of `06` §2 step 5(c): Fast-dLLM v2 7B, tau 0.9,
batch 1, block 32, `small_block_size = 32`, `use_block_cache = False`, **gen 512** (256
truncated 62 % of generations, `../phase0/fix512_L100/SUMMARY.md`), 0-shot chat template with
the boxed-answer instruction `v2/eval.py` builds.

Confidence is the loop's own `x1_p`, handed to the sink by the instrumented `batch_sample`.
"""
import argparse, json, os, re, sys, time, types
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import generation_functions                                             # noqa: E402
import profile_step_v2 as P                                             # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM            # noqa: E402
from dllm_skip.hook import install_skipping, uninstall_skipping, MODES  # noqa: E402
from dllm_skip.depth_schedule import (DepthSchedule, RegimeSchedule, StepState,   # noqa: E402
                                      leq_share)

MASK_ID = 151665
LAYERS = 28
ANS = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
BOXED = re.compile(r"boxed\{([^}]*)\}")


def extract(text):
    m = BOXED.findall(text)
    src = m[-1] if m else text
    g = ANS.findall(src)
    if not g:
        return None
    last = [t for t in g[-1] if t][-1]
    return last.replace("$", "").replace(",", "").rstrip(".")


class PassLog:
    def __init__(self, thr):
        self._pending, self.thr = [], thr

    def add(self, rec, conf, mask, committed):
        c = conf[mask]
        self._pending.append((rec, torch.stack(
            [c.mean(), c.min(), (c >= self.thr).sum().to(c.dtype),
             committed.sum().to(c.dtype)]) if c.numel() else None))

    def drain(self):
        vals = [None if t is None else t.tolist() for _, t in self._pending]
        for (rec, _), v in zip(self._pending, vals):
            if v is not None:
                rec["conf_mean"], rec["conf_min"], rec["n_ge_tau"], rec["committed"] = v
        self._pending = []


def run_cell(model, tok, prompts, golds, ids, sched, ctrl, mode, a, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sp = os.path.join(out_dir, "summary.json")
    if os.path.exists(sp):
        print(f">>> SKIP {out_dir}", flush=True); return
    if ctrl is not None:
        ctrl.mode = mode
    per, sf, plog = [], open(os.path.join(out_dir, "steps.jsonl"), "w"), PassLog(a.threshold)
    for n, (text, g, pid) in enumerate(zip(prompts, golds, ids)):
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
                schedule=sched, controller=ctrl, log=log, sink=plog)
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
        gen = tok.decode(out[0][L:], skip_special_tokens=True)
        pred = extract(gen)
        try:
            ok = pred is not None and abs(float(pred) - float(g)) < 1e-6
        except ValueError:
            ok = False
        nfe = len(log)
        per.append(dict(problem=int(pid), correct=int(ok), wall_s=wall, nfe=nfe, pred=pred,
                        gold=g, layer_steps=sum(r["depth"] for r in log)))
        plog.drain()
        for r in log:
            r["problem"] = int(pid); sf.write(json.dumps(r) + "\n")
        if (n + 1) % 50 == 0:
            print(f"    {n+1}/{len(prompts)}", flush=True)
    sf.close()
    nl = ctrl.n_layers if ctrl is not None else 28
    summ = dict(mode=mode, n=len(per), accuracy=sum(p["correct"] for p in per) / len(per),
                wall_s=sum(p["wall_s"] for p in per), nfe=sum(p["nfe"] for p in per),
                layer_steps=sum(p["layer_steps"] for p in per),
                full_layer_steps=sum(p["nfe"] for p in per) * nl,
                budget=None if sched is None else sched.budget,
                # PREREG 0.4 §2: the regime and its boundary travel with the cell. A regime
                # cell and the static cell it is compared with differ in exactly one field, and
                # this is it -- reading them apart from the directory name would be a guess.
                regime=getattr(sched, "regime", "static"),
                r_star=getattr(sched, "r_star", None),
                tau_w=a.threshold, tau_r=a.tau_r,
                args=vars(a), per_problem=per)
    # Raw layer count, then the byte-weighted L_eq ratio the gate axis is defined in:
    # a skipped `no-attn` layer still runs its FFN, so counting it as zero understates the
    # cost badly (0.10 instead of ~0.87 in Family B).
    fls = summ["full_layer_steps"]
    summ["depth_ratio_layers"] = summ["layer_steps"] / fls if fls else float("nan")
    share = leq_share(mode, LAYERS)
    skipped_layer_steps = fls - summ["layer_steps"]
    summ["leq_removed"] = skipped_layer_steps * share
    summ["depth_ratio"] = 1.0 - (summ["leq_removed"] / fls) if fls else float("nan")
    json.dump(summ, open(sp, "w"), indent=1)
    print(f">>> DONE {out_dir}  acc={summ['accuracy']:.4f}  wall={summ['wall_s']:.1f}s  "
          f"nfe={summ['nfe']}  depth={summ['depth_ratio']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    ap.add_argument("--ids", default="/home1/hyunhoko/DLLM/results/phase0.25/ids.json")
    ap.add_argument("--calib", default="",
                    help="calibrate_depth_v2.py output; supplies the layer ranking")
    ap.add_argument("--kl-json",
                    default="/home1/hyunhoko/DLLM/results/phase0.25/klgreedy_B.json")
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
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model)

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
    q, ans = P._gsm8k(split)
    prompts, golds = [], []
    for i in E:
        text = f"Question: {q[i]}\nAnswer:".replace("Answer:", P.GSM8K_INSTRUCTION)
        prompts.append(tok.apply_chat_template([{"role": "user", "content": text}],
                                               add_generation_prompt=True, tokenize=False))
        golds.append(ans[i].split("####")[-1].strip().replace(",", ""))

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
    regimes = [r.strip() for r in a.regimes.split(",") if r.strip()]
    for r in regimes:
        assert r in ("early", "late", "static"), f"unknown regime {r!r}"
    if any(r != "static" for r in regimes):
        assert a.r_star is not None, "--r-star is required for a non-static regime (Phase 0.35)"
    print(f"E: {len(E)} problems from GSM8K-{split} ({a.ids}) | L={L} | regimes: {regimes} | "
          f"r* = {a.r_star} | tau_r = {a.tau_r} | cells: {a.cells}", flush=True)
    if a.tau_r is not None:
        a.out = os.path.join(a.out, f"tau{a.tau_r:g}")
    for spec in a.cells.split(","):
        spec = spec.strip()
        if spec == "full":
            sch = DepthSchedule.static(L, [order], 0, keep_first=1, keep_last=a.keep_last)
            # a full-depth run has no regime, so it is written once, outside the regime tree
            run_cell(model, tok, prompts, golds, E, sch, ctrl, "identity", a,
                     os.path.join(a.out, "full", "0")); continue
        mode, k = spec.split(":")
        # `identity-kl` takes its skip set from the KL-greedy search on S (`PREREG.md` §3)
        # instead of the cosine ranking -- the one cell that asks whether selection quality,
        # not budget size, moves the pass count.
        if mode == "identity-kl":
            kl = json.load(open(a.kl_json))
            klset = list(kl["kl_greedy_set"])
            assert len(klset) == int(k), f"kl set has {len(klset)} layers, cell asks {k}"
            kl_order = klset + [l for l in range(L) if l not in klset]
            sch = DepthSchedule.static(L, [kl_order], int(k), keep_first=1,
                                       keep_last=a.keep_last, no_consecutive=True)
            got = sorted(set(range(L)) - set(sch.active_layers(StepState(0.5))))
            assert got == sorted(klset), f"schedule picked {got}, not the KL set {sorted(klset)}"
            ctrl.reuse_m = None
            for regime in regimes:
                sc = sch if regime == "static" else \
                    RegimeSchedule(base=sch, regime=regime, r_star=a.r_star)
                run_cell(model, tok, prompts, golds, E, sc, ctrl, "identity", a,
                         os.path.join(a.out, regime, "identity-kl", str(k)))
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
