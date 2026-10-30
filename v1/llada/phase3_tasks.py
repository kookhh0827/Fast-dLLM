"""Phase 3 stage 3-0: the new tasks of `stage2_runner.py --task` (results/phase3/PREREG.md, amendment A6).

The settings are those of results/phase3/references.md, section 1. The prompts are copied from the public evaluation
code that each reference used:

- math500: HuggingFaceH4/MATH-500 (500 problems), 0-shot. The system prompt of d1 (github.com/dllm-reasoning/d1 at
  faf85d6, eval/math500.py, MATH500_SYSTEM_PROMPT, copied as it is, with its stray quote) and the question form one user
  message (eval/gsm8k.py, create_prompt: system prompt + "\\n\\n" + question). The prompt ends with "<reasoning>", as
  d1 adds it (add_reasoning=True). Scoring: the last \\boxed{} of the output, against the reference answer, with
  math_verify.
- humanevalplus, mbppplus: the EvalPlus test sets (evalplus/humanevalplus, 164; evalplus/mbppplus, 378). The prompt of
  the Qwen2.5-Coder instruct suite (github.com/QwenLM/Qwen2.5-Coder at 33bc6aa, qwencoder-eval/instruct/eval_plus/
  model.py, class ChatML): the system and user messages under the model's chat template, and the answer starts with
  "```python\\n". The output stops at the first "\\n```" (the suite's extra stop string). The task prompt is the
  EvalPlus prompt, stripped: for HumanEval+ the dataset prompt; for MBPP+ '\"\"\"\\n<text>\\n<first test>\\n\"\"\"'.
  The solution is the sanitized output (`sanitize.py`, with the entry point), as `evalplus.sanitize` does in the suite.
  The tests run later, on a CPU (scripts/analysis/phase3_score.py).
- bigcodebench: bigcode/bigcodebench v0.1.4, split instruct, 1,140 tasks. The chat prompt of the official harness
  (bigcodebench v0.2.4, provider/utility.py make_raw_chat_prompt, with prefill, and the default prefixes of
  generate.py). The output stops at the first "\\n```". The harness sanitizes and runs it later, in its container.

- gsm8k_lmeval (gate C2 of the compressions, references.md section 4.1): GSM8K test (1,319) with the lm-eval 0.4.8
  task `gsm8k`: its prompt ("Question: ...\nAnswer:"), its few-shot examples from the train split (fewshot seed 1234,
  as lm-eval's simple_evaluate), --n-shot of them (4 for C2); the whole context is one user message under the chat
  template, as P4. Its stop strings. Scoring: lm-eval's exact_match after its two filters, "flexible-extract" (the
  `correct` field) and "strict-match" (also recorded).

Each task gives: the chat prompts, the documents (one per problem, in a fixed order) and the problem ids (the index in
the dataset). The dataset revisions are pinned below and go to load_dataset (amendment A14, change 5); the jobs run
with HF_HUB_OFFLINE=1. task_record() gives the data set, its revision, the prompt style and the checked inputs.

The exact prompts (amendment A14, change 5; EXACT_PROMPTS = True, for the models after LLaDA-8B-Instruct):
- math500: the system prompt of d1 has a space after the stray quote. The fork left it out, and the cells and gates of
  LLaDA-8B-Instruct keep that prompt (a deviation). check_d1_math500() compares the bytes with eval/math500.py of d1.
- mbppplus: the task text is the prompt of the EvalPlus release (MbppPlus v0.2.0, MbppPlus.jsonl.gz), stripped. The
  Hugging Face field keeps a source URL in some prompts. mbpp_release() checks all 378 prompts: the same task ids in
  the same order, the same entry points, and each prompt is the Hugging Face prompt or differs from it by a URL.
"""
import hashlib
import os
import re

TASKS = ("math500", "humanevalplus", "mbppplus", "bigcodebench", "gsm8k_lmeval")
CODE_TASKS = ("humanevalplus", "mbppplus", "bigcodebench")

# The revisions of the datasets in the cache (huggingface_hub.dataset_info on 2026-10-13).
DATA = {
    "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test"),
    "humanevalplus": ("evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b", "test"),
    "mbppplus": ("evalplus/mbppplus", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077", "test"),
    "bigcodebench": ("bigcode/bigcodebench", "b74c0d0bf70d2c0bc459be537895cca163007f1a", "v0.1.4"),
}
GSM8K_DATA = ("gsm8k", "740312add88f781978c0658806c59bc2815b9866")     # the parquet files that profile_step reads

# The inputs of the exact prompts (outside both repositories, with their sha256)
INPUTS = os.environ.get("DLLM_INPUTS", "/project2/ppanda_1750/kook/dllm/inputs")
D1_MATH500_FILE = ("d1_faf85d6/math500.py", "99228831a1567e936e26339ecaa121edea3dcc36657d5a688bf61bfeadf48d46")
MBPP_RELEASE_FILE = ("mbppplus_v0.2.0/MbppPlus.jsonl.gz",
                     "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63")
EXACT_PROMPTS = False                       # the runners set it (the v2 runner always; the v1 runner with a flag)
CHECKS = {}                                 # the results of the input checks, for task_record()

# d1 (faf85d6), eval/math500.py: copied as it is.
D1_MATH500_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"
"""

D1_MATH500_SYSTEM_PROMPT_EXACT = D1_MATH500_SYSTEM_PROMPT.replace('</answer>"\n', '</answer>" \n')

# Qwen2.5-Coder (33bc6aa), qwencoder-eval/instruct/eval_plus/model.py, class ChatML.
QWEN_SYSTEM = "You are an intelligent programming assistant to produce Python algorithmic solutions"
QWEN_USER = "Can you complete the following Python function?\n```python\n{prompt}\n```\n"
QWEN_PREFILL = "```python\n"

# bigcodebench v0.2.4: generate.py defaults and provider/utility.py.
BCB_INSTRUCTION = ("Please provide a self-contained Python script that solves the following problem in a markdown "
                   "code block:")
BCB_RESPONSE = ("Below is a Python script with a self-contained function that solves the problem and passes "
                "corresponding tests:")
BCB_SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-"

CODE_STOP = "\n```"

# Phase 3 amendment A10, change 1: gate T1 on MBPP+ and BigCodeBench runs again with the template of DiffuCoder
# (arXiv 2506.20639v2, Table 6, p. 24), under the chat template of the model: the system message, the task text
# between two lines "```", and the answer starts with "Here is the code to solve this problem:" and "```python".
# The task text is that of the default prompt: the EvalPlus prompt with its first test (MBPP+, as task_prompt) and
# instruct_prompt (BigCodeBench). The cells keep the default prompts (A10, change 3).
PROMPT_STYLES = ("default", "diffucoder6")
PROMPT_STYLE = "default"
DIFFUCODER_SYSTEM = "You are a helpful assistant."
DIFFUCODER_PREFILL = "Here is the code to solve this problem:\n```python\n"


def _load(task):
    from datasets import load_dataset
    name, rev, split = DATA[task]
    return list(load_dataset(name, split=split, revision=rev))


def _input(rel_sha):
    """The path of a checked input; stops if its sha256 is not the pinned one."""
    rel, sha = rel_sha
    path = os.path.join(INPUTS, rel)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if h != sha:
        raise SystemExit(f"{path}: sha256 {h}, not the pinned {sha}")
    return path


def check_d1_math500():
    """The bytes of MATH500_SYSTEM_PROMPT in eval/math500.py of d1 (faf85d6) against the exact prompt."""
    import ast
    path = _input(D1_MATH500_FILE)
    tree = ast.parse(open(path, encoding="utf-8").read())
    val = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
           and any(getattr(t, "id", None) == "MATH500_SYSTEM_PROMPT" for t in n.targets)]
    if len(val) != 1 or val[0].encode() != D1_MATH500_SYSTEM_PROMPT_EXACT.encode():
        raise SystemExit(f"the d1 MATH-500 system prompt in {path} is not the exact prompt of the fork")
    CHECKS["math500"] = dict(file=path, sha256=D1_MATH500_FILE[1], bytes_equal=True,
                             old_prompt_differs=D1_MATH500_SYSTEM_PROMPT != D1_MATH500_SYSTEM_PROMPT_EXACT)


_MBPP_RELEASE = {}


def mbpp_release(docs):
    """The prompts of the EvalPlus release by task id, after the check of all 378 prompts against the Hugging Face
    documents `docs` (all of them, in order)."""
    if _MBPP_RELEASE:
        return _MBPP_RELEASE
    import gzip
    import json
    path = _input(MBPP_RELEASE_FILE)
    rows = [json.loads(x) for x in gzip.open(path, "rt", encoding="utf-8") if x.strip()]
    if [r["task_id"] for r in rows] != [f"Mbpp/{d['task_id']}" for d in docs] or len(rows) != 378:
        raise SystemExit(f"{path}: the task ids are not those of the {len(docs)} Hugging Face problems")
    url = []
    for r, d in zip(rows, docs):
        if r["entry_point"] != mbpp_entry_point(d):
            raise SystemExit(f"{r['task_id']}: entry point {r['entry_point']} != {mbpp_entry_point(d)}")
        rel, hf = r["prompt"].strip(), f'"""\n{d["prompt"]}\n{d["test_list"][0]}\n"""'
        if not (rel.startswith('"""\n') and rel.endswith('\n"""')):
            raise SystemExit(f"{r['task_id']}: the release prompt is not a docstring")
        if rel != hf:
            if "http" not in hf:
                raise SystemExit(f"{r['task_id']}: the release prompt differs from the Hugging Face prompt, and not by "
                                 f"a URL")
            url.append(r["task_id"])
    _MBPP_RELEASE.update({int(r["task_id"].split("/")[1]): r["prompt"].strip() for r in rows})
    CHECKS["mbppplus"] = dict(file=path, sha256=MBPP_RELEASE_FILE[1], prompts=len(rows), differ_by_url=url)
    return _MBPP_RELEASE


def task_record(task):
    """The record of the task of a cell (amendment A14, change 5; message 2 of 2026-10-28, item 24)."""
    if task in DATA:
        name, rev, split = DATA[task]
    else:
        (name, rev), split = GSM8K_DATA, None
    return dict(task=task, dataset=name, revision=rev, split=split, prompt_style=PROMPT_STYLE,
                exact_prompts=EXACT_PROMPTS, checks=CHECKS.get(task))


def mbpp_entry_point(doc):
    """The function of the MBPP+ reference code that the first test calls (one for each of the 378 problems)."""
    defs = re.findall(r"^def\s+([A-Za-z_]\w*)\s*\(", doc["code"], re.M)
    hit = [d for d in defs if re.search(r"\b%s\s*\(" % re.escape(d), doc["test_list"][0])]
    assert len(hit) == 1, (doc["task_id"], defs)
    return hit[0]


def task_prompt(task, doc):
    """The task text inside the prompt (EvalPlus: the stripped EvalPlus prompt)."""
    if task == "humanevalplus":
        return doc["prompt"].strip()
    if task == "mbppplus":
        if EXACT_PROMPTS:
            return _MBPP_RELEASE[int(doc["task_id"])]
        return f'"""\n{doc["prompt"]}\n{doc["test_list"][0]}\n"""'
    raise ValueError(task)


def diffucoder6_prompt(tok, task, doc):
    text = task_prompt(task, doc) if task == "mbppplus" else doc["instruct_prompt"].strip()
    msg = [{"role": "system", "content": DIFFUCODER_SYSTEM}, {"role": "user", "content": f"```\n{text}\n```"}]
    return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) + DIFFUCODER_PREFILL


def chat_prompt(tok, task, doc):
    if PROMPT_STYLE == "diffucoder6":
        assert task in ("mbppplus", "bigcodebench"), f"the template of DiffuCoder Table 6 is for gate T1 only: {task}"
        return diffucoder6_prompt(tok, task, doc)
    if task == "math500":
        sp = D1_MATH500_SYSTEM_PROMPT_EXACT if EXACT_PROMPTS else D1_MATH500_SYSTEM_PROMPT
        msg = [{"role": "user", "content": sp + "\n\n" + doc["problem"]}]
        return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) + "<reasoning>"
    if task in ("humanevalplus", "mbppplus"):
        msg = [{"role": "system", "content": QWEN_SYSTEM},
               {"role": "user", "content": QWEN_USER.format(prompt=task_prompt(task, doc))}]
        return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) + QWEN_PREFILL
    if task == "bigcodebench":
        user = f"{BCB_INSTRUCTION}\n{doc['instruct_prompt'].strip()}\n"
        resp = f"{BCB_RESPONSE}\n```python\n{BCB_SPLITTER}\n```\n"
        return tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": resp}],
                                       tokenize=False).split(BCB_SPLITTER)[0]
    raise ValueError(task)


GSM8K_IGNORE = [",", "\\$", "(?s).*#### ", "\\.$"]


def _lmeval_norm(s):
    for r in GSM8K_IGNORE:
        s = re.sub(r, "", s)
    return s.lower()


def gsm8k_scores(resp, answer):
    """lm-eval gsm8k: (flexible-extract correct, strict-match correct, flexible prediction)."""
    flex = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", resp)
    flex = [g for g in flex[-1] if g][0] if flex else "[invalid]"
    strict = re.findall(r"#### (\-?[0-9\.\,]+)", resp)
    strict = strict[0] if strict else "[invalid]"
    ref = _lmeval_norm(answer)
    return _lmeval_norm(flex) == ref, _lmeval_norm(strict) == ref, flex


def build_gsm8k_lmeval(tok, n_shot, limit=0):
    from lm_eval.tasks import TaskManager, get_task_dict
    task = get_task_dict(["gsm8k"], TaskManager())["gsm8k"]
    task.set_config(key="num_fewshot", value=n_shot)
    task.set_fewshot_seed(seed=1234)
    task.build_all_requests(limit=limit or None, rank=0, world_size=1)
    insts = sorted(task.instances, key=lambda i: i.doc_id)
    prompts = [tok.apply_chat_template([{"role": "user", "content": i.args[0]}], add_generation_prompt=True,
                                       tokenize=False) for i in insts]
    docs = [dict(i.doc, task_id=int(i.doc_id)) for i in insts]
    return prompts, docs, [int(i.doc_id) for i in insts], list(insts[0].args[1]["until"])


def build(tok, task, limit=0, n_shot=0):
    """The chat prompts, the documents, the problem ids and the stop strings of a task."""
    if task == "gsm8k_lmeval":
        return build_gsm8k_lmeval(tok, n_shot, limit)
    docs = _load(task)
    if EXACT_PROMPTS and task == "math500":
        check_d1_math500()
    if EXACT_PROMPTS and task == "mbppplus":
        mbpp_release(docs)
    if limit:
        docs = docs[:limit]
    prompts = [chat_prompt(tok, task, d) for d in docs]
    ids = list(range(len(docs)))
    until = [CODE_STOP] if task in CODE_TASKS else []
    return prompts, docs, ids, until


def response(tok, gen_ids, eos_pos, until):
    """The output text: the tokens before the first end token, decoded, and cut at the first stop string."""
    ids = gen_ids if eos_pos < 0 else gen_ids[:eos_pos]
    s = tok.decode(ids, skip_special_tokens=True)
    for stop in until:
        if stop in s:
            s = s.split(stop)[0]
    return s


def fence_position(tok, gen_ids, eos_pos, stop=CODE_STOP):
    """Phase 3 amendment A13, change 11: where the output closes its code fence. Returns the character position of the
    first stop string in the text before the first end token, and the index of the generated token that holds its first
    character (None, None if the text has no stop string). The passes after the block of this token decode text that
    the scorer discards (A14, change 6)."""
    ids = gen_ids if eos_pos < 0 else gen_ids[:eos_pos]
    c = tok.decode(ids, skip_special_tokens=True).find(stop)
    if c < 0:
        return None, None
    lo, hi = 0, len(ids)
    while lo < hi:                      # the first prefix whose text is longer than c characters
        mid = (lo + hi) // 2
        if len(tok.decode(ids[:mid + 1], skip_special_tokens=True)) > c:
            hi = mid
        else:
            lo = mid + 1
    return c, lo


def last_boxed(text):
    """The content of the last \\boxed{...} (or \\fbox{...}), with matched braces; None if there is none."""
    i = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def math_correct(resp, answer):
    """MATH-500: the last boxed answer of the output against the reference answer, with math_verify."""
    from math_verify import parse, verify
    pred = last_boxed(resp)
    if pred is None or not pred.strip():
        return False, None
    try:
        return bool(verify(parse("\\boxed{" + answer + "}"), parse("\\boxed{" + pred + "}"))), pred
    except Exception:
        return False, pred


def solution(task, doc, resp):
    """EvalPlus: the sanitized output (the code that the tests run). BigCodeBench: the raw output (the harness
    sanitizes it)."""
    if task == "bigcodebench":
        return resp
    from sanitize import sanitize
    entry = doc["entry_point"] if task == "humanevalplus" else mbpp_entry_point(doc)
    return sanitize(resp, entry)


def sample_record(task, pid, doc, resp):
    rec = dict(problem=int(pid), task_id=doc["task_id"] if "task_id" in doc else doc.get("unique_id"), response=resp)
    if task in CODE_TASKS:
        rec["solution"] = solution(task, doc, resp)
    return rec
