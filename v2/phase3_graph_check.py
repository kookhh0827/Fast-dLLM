"""Phase 3 amendment A25 (DLLM): a test of the graph stack (phase3_graphs.py) against the eager passes, before the
identity check. No reading.

For the first --n GSM8K test problems (the prompt of stage2_runner_v2.py), with the settings of A7 and tau_r --tau-r,
the script decodes each problem eager and then on the graph stack, with the decode function of the runner. For each
problem it gives:
- the NFE and the output tokens of both (the same, or the first position that differs);
- the first pass whose logits are not the same bit for bit, the largest logit difference in that pass, and the first
  pass whose argmax differs;
- the time of a refinement pass and of a cache-writing pass (mean, ms) on each stack, and the record of the graphs.
The output JSON is written whole. Exit 1 if a problem differs.

    python phase3_graph_check.py --model-revision 0661abf --n 3 --out <file.json> [--compress gptq4 --compress-dir D
        --kernel-row-max 32 | --compress fp8 | --kl-json <M3 set file> (the layer removal of 6 layers)]
"""
import argparse, json, os, sys, types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage2_runner_v2 as SR                                           # noqa: E402  (its imports set the paths)
import phase3_graphs as PG                                              # noqa: E402
from stage2_runner_v2 import R, P, generation_functions                 # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM            # noqa: E402
from dllm_skip.hook import install_skipping                             # noqa: E402
from dllm_skip.depth_schedule import DepthSchedule, StepState           # noqa: E402


def recorder(model, inner, store):
    """model.forward that keeps the logits of each pass (on the GPU)."""
    def f(**kw):
        out = inner(**kw)
        store.append(out.logits.detach().clone())
        return out
    model.forward = f


def run(model, tok, text, sched, ctrl, a, store):
    timer = R.PassTimer()
    out, L, log, wall = SR.decode(model, tok, text, sched, ctrl, a, SR.PassLog(a.bar, a.eos_ids, model.device), timer)
    t, cnt, calls = R.pass_stats(timer)
    return dict(tokens=out[L:].tolist(), nfe=sum(1 for r in log if r["kind"] != "prompt"), wall=wall,
                ms={k: 1000 * t[k] / cnt[k] for k in ("refine", "cache_write") if cnt.get(k)}, calls=calls,
                kinds=[r["kind"] for r in log])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    ap.add_argument("--model-revision", required=True)
    ap.add_argument("--ids", default="/home1/hyunhoko/DLLM/results/phase0.3/ids_test.json")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tau-r", type=float, default=0.9)
    ap.add_argument("--compress", default=None, choices=["gptq4", "fp8", "awq4"])
    ap.add_argument("--compress-dir", default=None)
    ap.add_argument("--kernel-row-max", type=int, default=32)
    ap.add_argument("--kl-json", default=None, help="the layer removal (identity-kl:6) with this set file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.block_size, a.small_block_size, a.max_new_tokens = SR.A7["block_size"], SR.A7["small_block_size"], \
        SR.A7["max_new_tokens"]
    a.threshold, a.bar = 0.9, a.tau_r
    torch.manual_seed(0)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.model_revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.model_revision, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    model.mdm_sample = types.MethodType(generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model)
    a.eos_ids = sorted({t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                                    tok.convert_tokens_to_ids("<|endoftext|>"))
                        if isinstance(t, int) and t != tok.unk_token_id})
    comp = None
    if a.compress:
        import phase3_compress as PCX
        comp = PCX.apply(model, a.compress, a.compress_dir, row_max=a.kernel_row_max, model_name=a.model,
                         revision=a.model_revision)
    q, _ = P._gsm8k("test")
    E = json.load(open(a.ids))["E"][:a.n]
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"Question: {q[i]}\nAnswer:".replace(
        "Answer:", P.GSM8K_INSTRUCTION)}], add_generation_prompt=True, tokenize=False) for i in E]
    ctrl = install_skipping(model)
    L = ctrl.n_layers
    if a.kl_json:
        klset, rules, _ = SR.kl_set(a.kl_json, L)
        order = klset + [l for l in range(L) if l not in klset]
        sched = DepthSchedule.static(L, [order], len(klset), keep_first=rules["keep_first"],
                                     keep_last=rules["keep_last"], no_consecutive=rules["no_consecutive"])
        sched.expect_skip = set(klset)
    else:
        sched = DepthSchedule.static(L, [list(range(1, L - 1))], 0, keep_first=1, keep_last=8)
        sched.expect_skip = set()
    ctrl.mode = "identity"
    eager_fwd = model.forward
    res = dict(model_revision=a.model_revision, compression=a.compress, kl_json=a.kl_json, tau_r=a.tau_r,
               gpu=torch.cuda.get_device_name(0), torch=torch.__version__, run=R.run_record(), problems=[])
    # eager first (the warm-up problem once, as the runner does), then the graph stack
    run(model, tok, prompts[0], sched, ctrl, a, [])
    eager = []
    for text in prompts:
        st = []
        recorder(model, eager_fwd, st)
        eager.append((run(model, tok, text, sched, ctrl, a, st), st))
        del model.forward
    lens = [tok([t], return_tensors="pt").input_ids.shape[1] for t in prompts]
    gs = PG.GraphStack(model, ctrl, sched, lens, a.max_new_tokens)
    graph_fwd = model.forward
    model.forward = graph_fwd
    run(model, tok, prompts[0], sched, ctrl, a, [])
    ok = True
    for i, (text, (e, est)) in enumerate(zip(prompts, eager)):
        gst = []
        recorder(model, graph_fwd, gst)
        g = run(model, tok, text, sched, ctrl, a, gst)
        model.forward = graph_fwd
        same = e["tokens"] == g["tokens"] and e["nfe"] == g["nfe"]
        first_tok = next((j for j, (x, y) in enumerate(zip(e["tokens"], g["tokens"])) if x != y), None)
        first_logit = first_arg = None
        dmax = 0.0
        for j, (x, y) in enumerate(zip(est, gst)):
            if x.shape != y.shape:
                first_logit = j if first_logit is None else first_logit
                break
            d = float((x.float() - y.float()).abs().max())
            if d > 0 and first_logit is None:
                first_logit, dmax = j, d
            if first_arg is None and not torch.equal(x.argmax(-1), y.argmax(-1)):
                first_arg = j
            if first_logit is not None and first_arg is not None:
                break
        p = dict(problem=E[i], prompt_tokens=lens[i], same=same, nfe_eager=e["nfe"], nfe_graph=g["nfe"],
                 tokens_eager=len(e["tokens"]), tokens_graph=len(g["tokens"]), first_token_diff=first_tok,
                 first_pass_logits_diff=first_logit, max_logit_diff_in_that_pass=dmax, first_pass_argmax_diff=first_arg,
                 first_diff_pass_kind=None if first_logit is None else e["kinds"][first_logit],
                 ms_eager=e["ms"], ms_graph=g["ms"], wall_eager=e["wall"], wall_graph=g["wall"],
                 calls_eager=e["calls"], calls_graph=g["calls"])
        print(json.dumps(p), flush=True)
        res["problems"].append(p)
        ok = ok and same and first_logit is None
        del est, gst
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
