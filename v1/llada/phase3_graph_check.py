"""Phase 3 amendment A25 (DLLM): a test of the graph stack of LLaDA-8B-Instruct (phase3_graphs.py) against the eager
passes, before the check of A25. No reading.

For the first --n GSM8K test problems (the prompt of stage2_runner.py: 5 shots), with the settings of Block 1 (block
32, 256 new tokens, tau_w 0.9, tau_r --tau-r), the script decodes each problem eager and then on the graph stack, with
the decode function of the runner. With --aa it also decodes each problem eager a second time (the floor of the noise
between runs, A14, change 1). For each problem it gives:
- the NFE and the output tokens of each run (the same, or the first position that differs);
- the first pass whose logits are not the same bit for bit, the largest logit difference in that pass, and the first
  pass whose argmax differs. The logits are those that the sampler reads: the block of a cache-writing pass, all 32
  rows of a refinement pass;
- the time of a refinement pass and of a cache-writing pass (mean, ms) on each stack, the kernel-path calls, and the
  record of the graphs.
The output JSON is written whole. Exit 1 if a problem differs between eager and the graph stack.

    python phase3_graph_check.py --model-revision <rev> --n 3 --out <file.json> [--aa] [--gptq D --marlin-row-max 32 |
        --compress fp8 | --compress awq4 --marlin-row-max 32 | --kl-json <set file> (the layer removal of 7 layers)]
"""
import argparse, json, os, sys, types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage2_runner as SR                                              # noqa: E402  (its imports set the paths)
import phase3_graphs as PG                                              # noqa: E402
from stage2_runner import R, LLaDAModelLM                               # noqa: E402
from transformers import AutoTokenizer                                  # noqa: E402
from dllm_skip.hook import install_skipping                             # noqa: E402
from dllm_skip.depth_schedule import DepthSchedule, StepState           # noqa: E402


def recorder(model, inner, store, Lp, blk):
    """model.forward that keeps the logits that the sampler reads (a clone, on the GPU)."""
    n_cw = [0]

    def f(*args, **kw):
        out = inner(*args, **kw)
        if kw.get("past_key_values") is None:
            s = Lp + blk * n_cw[0]
            n_cw[0] += 1
            store.append(("cache_write", out.logits[:, s:s + blk].detach().clone()))
        else:
            store.append(("refine", out.logits.detach().clone()))
        return out
    model.forward = f


def run(model, inp, a, sched, ctrl):
    d = SR.decode_batch(model, inp, None, a, sched, ctrl, SR.PassLog(a.threshold))
    Lp = inp.shape[1]
    return dict(tokens=d["out"][0, Lp:].tolist(), nfe=int(d["st"]), wall=d["wall"],
                ms={k: 1000 * d["t"][k] / d["n"][k] for k in ("refine", "cache_write") if d["n"].get(k)},
                calls=d["calls"])


def compare(x, y):
    first_logit = first_arg = None
    dmax = 0.0
    for j, ((kx, lx), (ky, ly)) in enumerate(zip(x, y)):
        if kx != ky or lx.shape != ly.shape:
            return j, None, j
        d = float((lx.float() - ly.float()).abs().max())
        if d > 0 and first_logit is None:
            first_logit, dmax = j, d
        if first_arg is None and not torch.equal(lx.argmax(-1), ly.argmax(-1)):
            first_arg = j
        if first_logit is not None and first_arg is not None:
            break
    if first_logit is None and len(x) != len(y):
        first_logit = first_arg = min(len(x), len(y))
    return first_logit, dmax, first_arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--model-revision", required=True)
    ap.add_argument("--ids", default="/home1/hyunhoko/DLLM/results/phase0.3/ids_test.json")
    ap.add_argument("--calib", default="/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tau-r", type=float, default=0.9)
    ap.add_argument("--aa", action="store_true", help="also decode each problem eager a second time")
    ap.add_argument("--gptq", default=None, help="GPTQ checkpoint folder (the Marlin kernel)")
    ap.add_argument("--compress", default=None, choices=["fp8", "awq4"])
    ap.add_argument("--compress-dir", default="/project2/ppanda_1750/kook/dllm/models/phase3")
    ap.add_argument("--marlin-row-max", type=int, default=32)
    ap.add_argument("--kl-json", default=None, help="the layer removal (identity-kl:7) with this set file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.gen_length, a.steps, a.block_length, a.threshold, a.dus_base = 256, 256, 32, 0.9, None
    torch.manual_seed(0)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.model_revision, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(a.model, revision=a.model_revision, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    if a.gptq:
        import gptq_dequant
        gptq_dequant.load_marlin(model, a.gptq, 1, row_max=a.marlin_row_max)
    if a.compress:
        import phase3_compress as PCX
        PCX.apply(model, a.compress, a.compress_dir, row_max=a.marlin_row_max, model_name=a.model,
                  revision=a.model_revision)
    E = json.load(open(a.ids))["E"][:a.n]
    prompts, _ = SR.build(tok, E, 5, "test", "train")
    L = json.load(open(a.calib))["n_layers"]
    ctrl = install_skipping(model)
    if a.kl_json:
        j = json.load(open(a.kl_json))
        # the default rules of the runner when the set file has none (stage2_runner.main)
        klset, rules = sorted(j["kl_greedy_set"]), j.get("rules", dict(keep_first=1, keep_last=8, no_consecutive=True))
        order = klset + [l for l in range(L) if l not in klset]
        sched = DepthSchedule.static(L, [order], len(klset), keep_first=int(rules["keep_first"]),
                                     keep_last=int(rules["keep_last"]), no_consecutive=bool(rules["no_consecutive"]))
        assert sorted(set(range(L)) - set(sched.active_layers(StepState(0.5)))) == klset
        sched.expect_skip = set(klset)
    else:
        order = json.load(open(a.calib))["global_order"]
        sched = DepthSchedule.static(L, [order], 0, keep_first=1, keep_last=8, no_consecutive=True)
        sched.expect_skip = set()
    ctrl.mode = "identity"
    eager_fwd = model.forward
    res = dict(model_revision=a.model_revision, gptq=a.gptq, compression=a.compress, kl_json=a.kl_json,
               tau_r=a.tau_r, gpu=torch.cuda.get_device_name(0), torch=torch.__version__, run=R.run_record(),
               problems=[])
    gib = lambda: f"allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB"           # noqa: E731
    print(f"[check] after load: {gib()}", flush=True)
    gs = PG.GraphStack(model, ctrl, sched, L, a.gen_length, a.block_length)
    graph_fwd = model.forward
    inps = [tok([t], return_tensors="pt").input_ids.to(dev) for t in prompts]
    # the warm-up problem once on each stack (as the runner does)
    gs.prepare(inps[0].shape[1])
    for fwd in (eager_fwd, graph_fwd):
        model.forward = fwd
        run(model, inps[0], a, sched, ctrl)
    print(f"[check] after the warm-up: {gib()}; {gs.record()}", flush=True)
    ok = True
    for i, inp in enumerate(inps):
        Lp = inp.shape[1]
        gs.prepare(Lp)
        stores, runs = {}, {}
        for name, fwd in (("eager", eager_fwd), ("graph", graph_fwd)) + ((("eager2", eager_fwd),) if a.aa else ()):
            stores[name] = []
            recorder(model, fwd, stores[name], Lp, a.block_length)
            runs[name] = run(model, inp, a, sched, ctrl)
        model.forward = graph_fwd
        p = dict(problem=E[i], prompt_tokens=Lp)
        for name in [n for n in runs if n != "eager"]:
            e, g = runs["eager"], runs[name]
            fl, dmax, fa = compare(stores["eager"], stores[name])
            ft = next((j for j, (x, y) in enumerate(zip(e["tokens"], g["tokens"])) if x != y), None)
            p[name] = dict(same=e["tokens"] == g["tokens"] and e["nfe"] == g["nfe"], nfe=g["nfe"],
                           first_token_diff=ft, first_pass_logits_diff=fl, max_logit_diff_in_that_pass=dmax,
                           first_pass_argmax_diff=fa,
                           first_diff_pass_kind=None if fl is None or fl >= len(stores["eager"])
                           else stores["eager"][fl][0])
        for name, r in runs.items():
            p[f"{name}_run"] = dict(nfe=r["nfe"], ms=r["ms"], wall=r["wall"], calls=r["calls"])
        print(json.dumps(p), flush=True)
        res["problems"].append(p)
        ok = ok and p["graph"]["same"] and p["graph"]["first_pass_logits_diff"] is None
        del stores
        torch.cuda.empty_cache()
    res["graphs"] = gs.record()
    res["passed"] = ok
    gs.close()
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(dict(passed=ok, graphs=res["graphs"])), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
