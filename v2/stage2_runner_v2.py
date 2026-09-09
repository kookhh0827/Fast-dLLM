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
from dllm_skip.depth_schedule import DepthSchedule, StepState, leq_share           # noqa: E402

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
                use_block_cache=False, threshold=a.threshold,
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
                budget=None if sched is None else sched.budget, args=vars(a), per_problem=per)
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
    ap.add_argument("--calib", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--small-block-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.9)
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

    E = json.load(open(a.ids))["E"]
    if a.limit:
        E = E[:a.limit]
    q, ans = P._gsm8k("test" if a.ids.endswith("test.json") else "train")
    prompts, golds = [], []
    for i in E:
        text = f"Question: {q[i]}\nAnswer:".replace("Answer:", P.GSM8K_INSTRUCTION)
        prompts.append(tok.apply_chat_template([{"role": "user", "content": text}],
                                               add_generation_prompt=True, tokenize=False))
        golds.append(ans[i].split("####")[-1].strip().replace(",", ""))

    ctrl = install_skipping(model)
    L = ctrl.n_layers
    # Family B has no cosine calibration of its own; `01` §1b makes the structural rules
    # hypotheses here, so a Stage 2 cell that violates them is a finding, not an exclusion.
    order = list(range(1, L - 1))
    print(f"E: {len(E)} | L={L} | cells: {a.cells}", flush=True)
    for spec in a.cells.split(","):
        spec = spec.strip()
        if spec == "full":
            sch = DepthSchedule.static(L, [order], 0, keep_first=1, keep_last=a.keep_last)
            run_cell(model, tok, prompts, golds, E, sch, ctrl, "identity", a,
                     os.path.join(a.out, "full", "0")); continue
        mode, k = spec.split(":"); k = int(k)
        assert mode in MODES, mode
        nc = mode in ("identity", "reuse")
        sch = DepthSchedule.static(L, [order], k, keep_first=1,
                                   keep_last=a.keep_last if nc else 0, no_consecutive=nc)
        run_cell(model, tok, prompts, golds, E, sch, ctrl, mode, a,
                 os.path.join(a.out, mode, str(k)))
    uninstall_skipping(model)


if __name__ == "__main__":
    main()
