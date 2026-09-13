"""Phase 1b A2 / A3 / A6 on Family B's Phase 1 bank — the v2 twin of `v1/llada/phase1b_rank.py`, on the
bank and cache replay of `phase1_tok_v2.py` (the bank is read from its state file, never rebuilt).
Commit sets use the sampler's own rule (bf16 softmax, strict >, forced argmax, shifted logits);
max-probabilities for A2 are fp32. A6 generates the 128 calibration problems at full depth with the
fork's sampler (tau 0.9, block 32, small block 32, max 512, 0-shot) and scores them with the v2 runner's
own extraction.
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import generation_functions                                             # noqa: E402
import profile_step_v2 as P                                             # noqa: E402
import phase1_tok_v2 as T                                               # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM            # noqa: E402
from dllm_skip.hook import install_skipping, uninstall_skipping         # noqa: E402

CLASSES = ["digit", "operator", "other"]


def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra -= ra.mean(); rb -= rb.mean()
    d = (ra.norm() * rb.norm())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    ap.add_argument("--calib-ids", default="/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json")
    ap.add_argument("--bank-state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-a6", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16).to(dev).eval()
    model.mdm_sample = types.MethodType(generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model)
    st = torch.load(a.bank_state, weights_only=False)
    bank, calls = st["bank"], st["calls"]
    if a.limit:
        bank = bank[:a.limit]
    L = model.config.num_hidden_layers
    layers = list(range(1, L - 1))
    ctrl = install_skipping(model)
    replay = T.CacheReplay(model, calls, 32)
    print(f"bank: {len(bank)} canvases (not rebuilt); L = {L}", flush=True)

    spear = np.full((len(bank), len(layers)), np.nan, np.float32)
    dshift = np.full((len(bank), len(layers)), np.nan, np.float32)
    flips, cls_cache = [], {}
    t0 = time.perf_counter()
    for ci, c in enumerate(bank):
        pkv = replay.get(c)
        ids = c["ids"].to(dev)
        mask = (ids == T.MASK_ID)
        with torch.no_grad():
            ref_lg = T.block_logits(model, ids, pkv)
        ti = T.commit(ref_lg, mask, a.threshold)[0]
        m = mask[0]
        if m.sum() < 1 or ti.sum() < 1:
            continue
        pf = torch.softmax(ref_lg[0].float(), -1)
        top2 = pf.topk(2, dim=-1)
        mp_full = top2.values[:, 0]
        ref_arg = top2.indices[:, 0]
        pos = ti.nonzero().squeeze(-1)
        for t in ref_arg[pos].tolist():
            if t not in cls_cache:
                cls_cache[t] = CLASSES.index(T.token_class(tok, t))
        for j, l in enumerate(layers):
            ctrl.arm(tuple(x for x in range(L) if x != l))
            with torch.no_grad():
                lg = T.block_logits(model, ids, pkv)[0].float()
            ctrl.arm(None)
            ps = torch.softmax(lg, -1); mp_skip = ps.max(-1).values
            if m.sum() >= 3:
                spear[ci, j] = spearman(mp_full[m], mp_skip[m])
            dshift[ci, j] = float((mp_skip[pos] - mp_full[pos]).mean())
            sarg = lg.argmax(-1)
            for p in pos.tolist():
                if int(sarg[p]) != int(ref_arg[p]):
                    flips.append((ci, l, int(sarg[p]) == int(top2.indices[p, 1]),
                                  float(top2.values[p, 0] - top2.values[p, 1]), p,
                                  cls_cache[int(ref_arg[p])], float(top2.values[p, 0])))
        if (ci + 1) % 200 == 0:
            print(f"  A2/A3: {ci+1}/{len(bank)} ({(time.perf_counter()-t0)/(ci+1):.2f}s/canvas)", flush=True)
    uninstall_skipping(model)
    res = dict(layers=np.array(layers), spearman=spear, dshift=dshift,
               flips=np.array(flips, dtype=np.float64) if flips else np.zeros((0, 7)),
               c_problem=np.array([c["problem"] for c in bank]), c_bucket=np.array([c["bucket"] for c in bank]),
               c_block=np.array([c["block"] for c in bank]))

    if not a.skip_a6:
        import stage2_runner_v2 as SR
        calib_ids = json.load(open(a.calib_ids))["calib_ids"][:max(c["problem"] for c in bank) + 1]
        q, ans = P._gsm8k("train")
        correct = []
        for n, i in enumerate(calib_ids):
            text = f"Question: {q[i]}\nAnswer:".replace("Answer:", P.GSM8K_INSTRUCTION)
            text = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                           tokenize=False)
            ids_t = tok([text], return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                out = model.mdm_sample(ids_t, tokenizer=tok, block_size=32, small_block_size=32,
                                       max_new_tokens=512, mask_id=T.MASK_ID, min_len=ids_t.shape[1],
                                       seq_len=torch.tensor([ids_t.shape[1]], device=dev), use_block_cache=False,
                                       threshold=a.threshold, schedule=None, controller=None, log=None)
            gen = tok.decode(out[0][ids_t.shape[1]:], skip_special_tokens=True)
            pred = SR.extract(gen)
            gold = ans[i].split("####")[-1].strip().replace(",", "")
            try:
                ok = pred is not None and abs(float(pred) - float(gold)) < 1e-6
            except ValueError:
                ok = False
            correct.append(int(ok))
            if (n + 1) % 32 == 0:
                print(f"  A6: {n+1}/{len(calib_ids)}, accuracy so far {sum(correct)/len(correct):.3f}", flush=True)
        res["calib_correct"] = np.array(correct)
    np.savez_compressed(a.out + ".npz", **res)
    print(f"written {a.out}.npz  flips {len(flips)}")


if __name__ == "__main__":
    main()
