"""Phase 2 P5 harness check (decision D7, section 4): the pad mask of the batched runner. Not a P5 cell.

For each of the first N problems of E, the same prompt runs four times (bf16, tau 0.9, DualCache, the runner's k = 0
schedule, as `stage2_runner.py --cells full`):
  1. batch 1, no mask (the old path);
  2. batch 1 again (the run noise of batch 1);
  3. a batch of 8 identical rows: no pads, so no mask;
  4. a batch of 8 identical rows, where row r has 5 r extra left pads that the pad mask hides as keys (row 0 has none).
For each row it records: the generated tokens equal to run 1 (and the first differing position), the extracted
answer, and the row's passes (the cache-writing passes plus the refinement passes where the row has a masked
position, as the runner counts them).
Family A is not bit-reproducible in bf16, so a row can differ from run 1. Rows of one batch with the same input
(run 3, all rows) must give the same output.

    python phase2_padcheck.py --n 4 --out <dir>/padcheck.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_runner as R                                               # noqa: E402  (sets sys.path for dllm_skip)
import generate as G                                                    # noqa: E402
from model.modeling_llada import LLaDAModelLM, set_pad_mask            # noqa: E402
from transformers import AutoTokenizer                                 # noqa: E402
from dllm_skip.hook import install_skipping                            # noqa: E402
from dllm_skip.depth_schedule import DepthSchedule                     # noqa: E402

B, PAD_STEP = 8, 5


def run(model, tok, ctrl, sched, rows, extra, eos_t):
    """One batch: `rows` token lists, `extra[r]` more left pads on row r. Returns one record per row."""
    Lp = max(len(x) + e for x, e in zip(rows, extra))
    inp = torch.full((len(rows), Lp), tok.pad_token_id, dtype=torch.long)
    for r, x in enumerate(rows):
        inp[r, Lp - len(x):] = torch.tensor(x, dtype=torch.long)
    inp = inp.to(model.device)
    padded = any(len(x) != Lp for x in rows)
    if padded:
        pm = torch.ones((len(rows), Lp + 256), dtype=torch.bool, device=model.device)
        for r, x in enumerate(rows):
            pm[r, :Lp - len(x)] = False
        set_pad_mask(model, pm)
    plog, log = R.PassLog(0.9, rows=len(rows) > 1), []
    ctrl.mode = "identity"
    ctrl.new_block()
    with torch.no_grad():
        out, st = G.generate_with_dual_cache(model, inp, steps=256, gen_length=256, block_length=32, temperature=0.0,
                                             remasking="low_confidence", threshold=0.9, tau_r=0.9, schedule=sched,
                                             controller=ctrl, log=log, sink=plog, dus_base=None)
    if padded:
        set_pad_mask(model, None)
    plog.drain()
    recs = []
    for r in range(len(rows)):
        gen = out[r, Lp:]
        hit = torch.isin(gen, eos_t).nonzero()
        nfe = int(st) if len(rows) == 1 else sum(1 for p in log if p["cache_write"] or p["rows_masked"][r] > 0)
        recs.append(dict(tokens=gen.tolist(), pred=R.flexible_extract(tok.decode(gen, skip_special_tokens=True)),
                         nfe=nfe, eos_pos=int(hit[0, 0]) if hit.numel() else -1, pads=Lp - len(rows[r])))
    return recs


def compare(ref, rec):
    d = [i for i, (a, b) in enumerate(zip(ref["tokens"], rec["tokens"])) if a != b]
    return dict(same_tokens=not d, first_diff=d[0] if d else None, n_diff=len(d), same_pred=rec["pred"] == ref["pred"],
                pred=rec["pred"], nfe=rec["nfe"], pads=rec["pads"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--ids", default="/home1/hyunhoko/DLLM/results/phase0.25/ids.json")
    ap.add_argument("--calib", default="/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    j = json.load(open(a.ids))
    ids = j["E"][:a.n]                                                  # the E split, GSM8K train
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    eos_t = torch.tensor(sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")}), device="cuda")
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    cal = json.load(open(a.calib))
    sched = DepthSchedule.static(cal["n_layers"], [cal["global_order"]], 0, keep_first=1, keep_last=8,
                                 no_consecutive=True)
    ctrl = install_skipping(model)
    prompts, golds = R.build(tok, ids, 5, "train")
    res = []
    for pid, text, g in zip(ids, prompts, golds):
        x = tok(text).input_ids
        b1 = run(model, tok, ctrl, sched, [x], [0], eos_t)[0]
        b1b = run(model, tok, ctrl, sched, [x], [0], eos_t)[0]
        same = run(model, tok, ctrl, sched, [x] * B, [0] * B, eos_t)
        pad = run(model, tok, ctrl, sched, [x] * B, [PAD_STEP * r for r in range(B)], eos_t)
        rec = dict(problem=pid, gold=g, prompt_len=len(x), batch1=dict(pred=b1["pred"], nfe=b1["nfe"]),
                   batch1_again=compare(b1, b1b),
                   identical_rows=[compare(b1, r) for r in same],
                   identical_rows_equal_row0=all(r["tokens"] == same[0]["tokens"] for r in same),
                   padded_rows=[compare(b1, r) for r in pad],
                   padded_rows_equal_row0=[r["tokens"] == pad[0]["tokens"] for r in pad])
        res.append(rec)
        print(f"problem {pid}: batch 1 pred {b1['pred']} nfe {b1['nfe']}; batch 1 again same tokens "
              f"{rec['batch1_again']['same_tokens']}; identical rows: all rows equal {rec['identical_rows_equal_row0']}, "
              f"rows equal to batch 1 {sum(c['same_tokens'] for c in rec['identical_rows'])}/{B}; padded rows: equal to "
              f"row 0 {sum(rec['padded_rows_equal_row0'])}/{B}, equal to batch 1 "
              f"{sum(c['same_tokens'] for c in rec['padded_rows'])}/{B}, same answer "
              f"{sum(c['same_pred'] for c in rec['padded_rows'])}/{B}, row NFE {[c['nfe'] for c in rec['padded_rows']]}",
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
