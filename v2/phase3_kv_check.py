"""Phase 3 amendment A22: the check before the KV-cache cells of Fast-dLLM v2 7B (DLLM, results/phase3/PREREG.md).

SoloQ is private: this file holds no SoloQ code. phase3_records.soloq_state imports SoloQ from SOLOQ_PATH and applies
its patch from the environment of the job, as in each SoloQ cell of stage2_runner_v2.py.

On the first problem of the task (the prompt of stage2_runner_v2.py), the script
1. runs the pass over the prompt (a cache-writing pass: use_cache=True, update_past_key_values=True) on the bf16 model
   and keeps the stored keys and values of each layer;
2. applies SoloQ (soloq_state with soloq_cell=True), records its configuration, and runs the same pass again;
3. gives the relative error ||KV(SoloQ) - KV(bf16)||_F / ||KV(bf16)||_F of the stored keys and of the stored values of
   each layer. Layer 0 has the same input in both runs, so its error is the error of the KV quantizer alone. Each later
   layer also carries the error of the layers before it;
4. counts the calls of the SoloQ KV quantizers in one pass of each type of the sampler (generation_functions.py): the
   pass over the prompt, a refinement pass (update_past_key_values=False), a refinement pass on the block cache
   (use_block_cache=True) and a cache-writing pass of a block (update_past_key_values=True).

The check fails (exit 1) if a relative error of layer 0 is not above 0. The output JSON is written whole.

    python phase3_kv_check.py --task gsm8k --model-revision 0661abf --out <file.json>
"""
import argparse, json, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "v1", "llada")))
import phase3_tasks as P3                                               # noqa: E402
import phase3_records as R                                              # noqa: E402
import profile_step_v2 as P                                             # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM            # noqa: E402

BLOCK = 32


def prompt_of(tok, task):
    """The prompt of the first problem, as stage2_runner_v2.py builds it (A14, change 5: the exact prompts)."""
    if task == "gsm8k":
        q, _ = P._gsm8k("test")
        text = f"Question: {q[0]}\nAnswer:".replace("Answer:", P.GSM8K_INSTRUCTION)
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       tokenize=False)
    P3.EXACT_PROMPTS = True
    prompts, _, _, _ = P3.build(tok, task, 1)
    return prompts[0]


def kv_of(pkv, n_layers):
    return [(pkv[i][0].detach().float().clone(), pkv[i][1].detach().float().clone()) for i in range(n_layers)]


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    ap.add_argument("--model-revision", required=True)
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "math500"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.model_revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.model_revision, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    nl = model.config.num_hidden_layers
    ids = tok(prompt_of(tok, a.task), return_tensors="pt").input_ids.to(dev)
    L = ids.shape[1] // BLOCK * BLOCK
    assert L >= BLOCK, f"prompt of {ids.shape[1]} tokens"
    x, blk = ids[:, :L], ids[:, L - BLOCK:L]

    with torch.no_grad():
        ref = kv_of(model(input_ids=x, use_cache=True, update_past_key_values=True, block_size=BLOCK).past_key_values,
                    nl)
        rec = R.soloq_state(model, True)
        print(f"[soloq] {rec}", flush=True)
        calls = {}

        def counted(name, **kw):
            c0 = R.soloq_kv_counters(model)
            out = model(**kw)
            c1 = R.soloq_kv_counters(model)
            calls[name] = None if c0 is None else {k: c1[k] - c0[k] for k in c0}
            return out

        pkv = counted("prompt pass (cache-writing)", input_ids=x, use_cache=True, update_past_key_values=True,
                      block_size=BLOCK).past_key_values
        got = kv_of(pkv, nl)
        counted("refinement pass", input_ids=blk, use_cache=True, past_key_values=pkv, update_past_key_values=False)
        counted("refinement pass on the block cache", input_ids=blk, use_cache=True, past_key_values=pkv,
                update_past_key_values=False, use_block_cache=True)
        counted("cache-writing pass of a block", input_ids=blk, use_cache=True, past_key_values=pkv,
                update_past_key_values=True, block_size=BLOCK)

    ek = [rel(g[0], r[0]) for g, r in zip(got, ref)]
    ev = [rel(g[1], r[1]) for g, r in zip(got, ref)]
    res = dict(task=a.task, model_revision=a.model_revision, prompt_tokens=int(ids.shape[1]), prompt_pass_tokens=L,
               n_layers=nl, gpu=torch.cuda.get_device_name(0), torch=torch.__version__, run=R.run_record(), soloq=rec,
               rel_error_keys=ek, rel_error_values=ev,
               rel_error_layer0=dict(keys=ek[0], values=ev[0]),
               rel_error_max=dict(keys=max(ek), values=max(ev)),
               quantizer_calls=calls, layers=nl,
               passed=bool(ek[0] > 0 and ev[0] > 0))
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps({k: res[k] for k in ("rel_error_layer0", "rel_error_max", "quantizer_calls", "passed")}),
          flush=True)
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
