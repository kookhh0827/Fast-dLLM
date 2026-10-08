"""Phase 2 P4: HumanEval and MBPP for `stage2_runner.py --task` (`results/phase2/PREREG.md`, section 7).

Prompts. lm-eval's own request builder (lm_eval 0.4.8, as in `eval_llada.py`) with the task defaults: HumanEval
0-shot (164 problems), MBPP 3-shot (`first_n` examples, 500 test problems). As in `eval_llada.py` for an instruct
model, the whole context is one user message under the chat template.

Output text. As `eval_llada.py` cuts it: HumanEval keeps the whole decoded output (special tokens removed); the
other tasks cut the output at the first stop sequence of the task, then encode and decode it again without
special tokens.

Scoring. HumanEval: the rule of the fork's `postprocess_code.py` (the ```python block after the prompt, `sanitize`,
pass@1 by `code_eval`). MBPP: lm-eval's own metric (pass@1 by `code_eval` on the response). The generated code runs
in `code_eval`'s subprocesses with its timeout; this needs HF_ALLOW_CODE_EVAL=1.
"""
import os

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")


def build(tok, task_name, limit=0):
    """Returns the task object, the chat prompts, the docs, the doc ids and the stop sequences."""
    from lm_eval.tasks import TaskManager, get_task_dict
    task = get_task_dict([task_name], TaskManager())[task_name]
    task.build_all_requests(limit=limit or None, rank=0, world_size=1)
    insts = sorted(task.instances, key=lambda i: i.doc_id)
    prompts, docs, ids, until = [], [], [], None
    for inst in insts:
        ctx, gen_kwargs = inst.args
        prompts.append(tok.apply_chat_template([{"role": "user", "content": ctx}], add_generation_prompt=True,
                                               tokenize=False))
        docs.append(inst.doc)
        ids.append(int(inst.doc_id))
        until = list(gen_kwargs["until"])
    return task, prompts, docs, ids, until


def response(tok, gen_ids, task_name, until):
    if task_name == "humaneval":
        return tok.decode(gen_ids, skip_special_tokens=True)
    s = tok.decode(gen_ids, skip_special_tokens=False)
    for stop in until:
        if stop in s:
            s = s.split(stop)[0]
    return tok.decode(tok(s)["input_ids"], skip_special_tokens=True)


def score(task, task_name, docs, responses, workers=1):
    """pass@1 (0 or 1) for each problem, in the order of `docs`.

    One worker thread. `code_eval` forks a process for each program from its worker threads, and filelock >= 3.32
    raises "os.fork is unsafe while filelock is changing descriptor ownership" when two threads fork at the same
    time (job 6332522). With one worker, only one fork runs at a time.
    """
    import evaluate
    from sanitize import sanitize
    code_eval = evaluate.load("code_eval")
    refs = [task.doc_to_target(d) for d in docs]
    if task_name == "humaneval":
        preds = [[sanitize(d["prompt"] + "\n" + r.split("```python\n", 1)[-1].split("```")[0], d["entry_point"])]
                 for d, r in zip(docs, responses)]
    else:
        preds = [[r] for r in responses]
    _, res = code_eval.compute(references=refs, predictions=preds, k=[1], num_workers=workers)
    return [int(res[i][0][1]["passed"]) for i in range(len(docs))], preds
