"""Phase 2 week 2, gate G1'.1 (amendment A8 of results/phase2/PREREG_week2.md): the exact test of the pad mask.

For each of the first N problems of E, one batch holds the problem's prompt in its 8 rows. Before the prompt, each
row has 35 hidden positions. In row r (r = 0 to 7), the first 5 r of them hold the pad token, and the other 35 - 5 r
hold random ordinary tokens (a fixed seed, different in each row; never the mask token, the pad token, an end token
or another special token).
  * Test: the pad mask hides all 35 positions in each row. Pass: the 8 rows give the same tokens and the same row NFE.
  * Negative control: the pad mask hides only the pad tokens, so the random tokens are visible. The rows must
    differ in at least one problem; if not, the test gives no result.
This script makes only the rows and the pad mask. The decoding is `stage2_runner.decode_batch`, the code of P5
(bf16, tau 0.9, DualCache, the k = 0 schedule of `--cells full`). It also reads back the hidden positions of each row
from the mask that the blocks hold.

    python phase2_padcheck.py --n 8 --out <dir>/padcheck.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_runner as R                                               # noqa: E402  (sets sys.path for dllm_skip)
from model.modeling_llada import LLaDAModelLM                          # noqa: E402
from transformers import AutoTokenizer                                 # noqa: E402
from dllm_skip.hook import install_skipping                            # noqa: E402
from dllm_skip.depth_schedule import DepthSchedule                     # noqa: E402

ROWS, HIDDEN, STEP, SEED, MASK_ID = 8, 35, 5, 20261009, 126336


def ordinary_tokens(tok):
    banned = set(tok.all_special_ids) | set(tok.get_added_vocab().values()) | {MASK_ID, tok.pad_token_id,
                                                                               tok.eos_token_id,
                                                                               tok.convert_tokens_to_ids("<|eot_id|>")}
    return np.array([i for i in range(tok.vocab_size) if i not in banned])


def run_batch(model, tok, x, pid, hide_all, cand, args, sched, ctrl):
    rows = []
    for r in range(ROWS):
        rng = np.random.default_rng([SEED, pid, r])
        rows.append([tok.pad_token_id] * (STEP * r) + rng.choice(cand, size=HIDDEN - STEP * r).tolist() + x)
    inp = torch.tensor(rows, dtype=torch.long, device=model.device)
    pm = torch.ones((ROWS, inp.shape[1] + args.gen_length), dtype=torch.bool, device=model.device)
    for r in range(ROWS):
        pm[r, :HIDDEN if hide_all else STEP * r] = False
    d = R.decode_batch(model, inp, pm, args, sched, ctrl, R.PassLog(args.threshold, rows=True))
    toks = d["out"][:, inp.shape[1]:].tolist()
    nfe = [sum(1 for p in d["log"] if p["cache_write"] or p["rows_masked"][r] > 0) for r in range(ROWS)]
    diff = [next((k for k, (a, b) in enumerate(zip(toks[0], t)) if a != b), None) for t in toks]
    return dict(same_tokens=all(t == toks[0] for t in toks), same_nfe=len(set(nfe)) == 1, nfe=nfe,
                first_diff_from_row0=diff, hidden=[int(h) for h in d["hidden"]],
                hidden_expected=[HIDDEN if hide_all else STEP * r for r in range(ROWS)],
                pred=[R.flexible_extract(tok.decode(t, skip_special_tokens=True)) for t in d["out"][:, inp.shape[1]:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--ids", default="/home1/hyunhoko/DLLM/results/phase0.25/ids.json")
    ap.add_argument("--calib", default="/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    args = argparse.Namespace(steps=256, gen_length=256, block_length=32, threshold=0.9, tau_r=0.9, dus_base=None)
    ids = json.load(open(a.ids))["E"][:a.n]                            # the E split, GSM8K train
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    cand = ordinary_tokens(tok)
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    cal = json.load(open(a.calib))
    sched = DepthSchedule.static(cal["n_layers"], [cal["global_order"]], 0, keep_first=1, keep_last=8,
                                 no_consecutive=True)
    ctrl = install_skipping(model)
    ctrl.mode = "identity"
    prompts, _ = R.build(tok, ids, 5, "train")
    res = []
    for pid, text in zip(ids, prompts):
        x = tok(text).input_ids
        t = run_batch(model, tok, x, pid, True, cand, args, sched, ctrl)
        nc = run_batch(model, tok, x, pid, False, cand, args, sched, ctrl)
        res.append(dict(problem=pid, prompt_len=len(x), test=t, negative_control=nc))
        print(f"problem {pid}: test (all 35 hidden): same tokens {t['same_tokens']}, same row NFE {t['same_nfe']} "
              f"{t['nfe']}, hidden {t['hidden']}; negative control (pads hidden only): same tokens {nc['same_tokens']}, "
              f"row NFE {nc['nfe']}, hidden {nc['hidden']}", flush=True)
    out = dict(rows=ROWS, hidden=HIDDEN, step=STEP, seed=SEED, n_ordinary_tokens=int(len(cand)),
               test_pass=all(r["test"]["same_tokens"] and r["test"]["same_nfe"] for r in res),
               negative_control_differs=any(not r["negative_control"]["same_tokens"] for r in res),
               hidden_readback_ok=all(r[k]["hidden"] == r[k]["hidden_expected"] for r in res
                                      for k in ("test", "negative_control")),
               problems=res)
    print(f"G1'.1: test pass {out['test_pass']}; negative control differs {out['negative_control_differs']}; "
          f"hidden positions read back as built {out['hidden_readback_ok']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
