"""Phase 2 P7b, checks I1 and I2 of amendment A10 (results/phase2/PREREG_week2.md): do our two implementations do
what the authors' code does? One process, one model object (bf16), the first N problems of E, batch 1.

  * I1 (DC-Leap): the ported function (`generate.generate_with_dc_leap`, run by `stage2_runner.decode_batch` as a
    P7b cell runs it) and the authors' function (`llada/generate.py` of the DC-Leap repository) give the same
    tokens and the same NFE in every problem. The NFE of the authors' function is the count of model calls.
  * I2 (CAI-DLLM): our rule (`generate.CaiRule` in the DualCache sampler, block 64, run by `decode_batch`) calls a
    probe after each pass. The probe gives the same block logits to the authors' commit code
    (`cai_dllm/confidence_gating.py`: TokenBudgetManager, PositionAwareScheduler, GrindingPhaseDetector,
    confidence_gated_sample, with `apd_schedule.probe_step0_confidence` and the default table of
    `per_block_schedule`, as `--cai_mode apd_per_block` builds it). Both compute the confidences from these logits.
    Pass: the commit sets are the same at every pass. The end-token delay is off in the authors' code (eos_token -1).

The authors' code comes from scratch copies of the two repositories (paths as arguments). It is not in the fork.

    python phase2_p7check.py --dcleap-repo <DC-Leap> --cai-repo <CAI-DLLM> --n 8 --out <dir>/p7check.json
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_runner as R                                               # noqa: E402  (sets sys.path for dllm_skip)
from model.modeling_llada import LLaDAModelLM                          # noqa: E402
from transformers import AutoTokenizer                                 # noqa: E402
from dllm_skip.hook import install_skipping                            # noqa: E402
from dllm_skip.depth_schedule import DepthSchedule                     # noqa: E402

DCLEAP_COMMIT, CAI_COMMIT = "e105d45", "6246dad"
GEN, BLOCK_CAI = 256, 64


def repo_commit(path):
    return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


class AuthorsCAI:
    """The authors' commit code, driven as `cai_generate3.cai_batch_generate` drives it (one block: new budget
    manager and grind detector; step 0: step-0 confidence, budgets, position scheduler; after each step: the grind
    detector records the commits), on the logits of our pass."""

    def __init__(self, cg, aps, pbs, full_len, num_blocks):
        self.cg, self.aps, self.full_len = cg, aps, full_len
        self.table = pbs.PerBlockScheduleTable.default(num_blocks)
        self.passes, self.mismatch, self.token_mismatch, self.first = 0, 0, 0, None

    def __call__(self, nb, step, s, logits, masked, ours, x0_ours):
        cg, dev = self.cg, logits.device
        if step == 0:
            self.budget = cg.TokenBudgetManager(block_len=BLOCK_CAI)
            self.grind = cg.GrindingPhaseDetector()
            self.pos_sched = None
        pos = torch.arange(s, s + BLOCK_CAI, device=dev).unsqueeze(0)
        mask_index = torch.zeros((1, self.full_len), dtype=torch.bool, device=dev)
        mask_index[:, s:s + BLOCK_CAI] = masked
        all_conf = torch.zeros((1, self.full_len), dtype=torch.float32, device=dev)
        if step == 0:
            conf0 = self.aps.probe_step0_confidence(logits, mask_index, pos)
            self.budget.init_from_step0(conf0, mask_index, pos)
            self.pos_sched = cg.PositionAwareScheduler(self.table.get_scheduler(nb))
        x0, _, sel = cg.confidence_gated_sample(
            logits=logits, pos=pos, mask_index=mask_index, all_confidence=all_conf, step=step,
            total_steps=BLOCK_CAI, block_start=s, budget_mgr=self.budget, pos_scheduler=self.pos_sched,
            grind_detector=self.grind, eos_token=-1, temperature=0.0, exist_eos=None)
        theirs = torch.zeros_like(ours)
        theirs[0, sel[0]] = True
        self.grind.record_step(int(theirs.sum()), int(mask_index.sum()), step)
        self.passes += 1
        if not torch.equal(theirs, ours):
            self.mismatch += 1
            if self.first is None:
                self.first = dict(block=nb, step=step, ours=ours[0].nonzero().flatten().tolist(),
                                  theirs=theirs[0].nonzero().flatten().tolist())
        elif not torch.equal(x0[0][ours[0]], x0_ours[0][ours[0]]):
            self.token_mismatch += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--model-revision", default=None, help="recorded only, as in stage2_runner.py")
    ap.add_argument("--ids", default="/home1/hyunhoko/DLLM/results/phase0.25/ids.json")
    ap.add_argument("--calib", default="/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json")
    ap.add_argument("--dcleap-repo", required=True)
    ap.add_argument("--cai-repo", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    commits = dict(dc_leap=repo_commit(a.dcleap_repo), cai_dllm=repo_commit(a.cai_repo))
    assert commits == dict(dc_leap=DCLEAP_COMMIT, cai_dllm=CAI_COMMIT), f"scratch copies at {commits}"

    spec = importlib.util.spec_from_file_location("dcleap_authors", os.path.join(a.dcleap_repo, "llada", "generate.py"))
    dcl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dcl)
    sys.path.append(os.path.join(a.cai_repo, "cai_dllm"))               # last, so that it shadows none of our modules
    import confidence_gating as cg                                      # noqa: E402
    import apd_schedule as aps                                          # noqa: E402
    import per_block_schedule as pbs                                    # noqa: E402

    ids = json.load(open(a.ids))["E"][:a.n]                            # the E split, GSM8K train
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    cal = json.load(open(a.calib))
    sched = DepthSchedule.static(cal["n_layers"], [cal["global_order"]], 0, keep_first=1, keep_last=8,
                                 no_consecutive=True)
    ctrl = install_skipping(model)
    ctrl.mode = "identity"
    prompts, _ = R.build(tok, ids, 5, "train")
    base = dict(steps=256, gen_length=GEN, threshold=0.9, tau_r=None, dus_base=None, dc_commit=0.70, dc_draft=0.98,
                dc_window=128)
    args_dcl = argparse.Namespace(**base, block_length=32, sampler="dc_leap", cai=False)
    args_cai = argparse.Namespace(**base, block_length=BLOCK_CAI, sampler="dual", cai=True)
    calls = [0]
    i1, i2 = [], []
    for pid, text in zip(ids, prompts):
        inp = torch.tensor([tok(text).input_ids], dtype=torch.long, device=model.device)
        lp = inp.shape[1]
        # I1: the authors' function, then the port
        h = model.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
        calls[0] = 0
        with torch.no_grad():
            out_a = dcl.generate_with_dc_leap(model, inp, steps=256, commit_thres=0.70, draft_thres=0.98, gen_length=GEN,
                                              block_length=32, max_window_size=128, cfg_scale=0.0, temperature=0.0)
        h.remove()
        nfe_a = calls[0]
        d = R.decode_batch(model, inp, None, args_dcl, sched, ctrl, R.PassLog(0.9))
        ta, to = out_a[0, lp:].tolist(), d["out"][0, lp:].tolist()
        first = next((k for k, (x, y) in enumerate(zip(ta, to)) if x != y), None)
        i1.append(dict(problem=pid, same_tokens=ta == to, nfe_authors=nfe_a, nfe_port=int(d["st"]),
                       same_nfe=nfe_a == int(d["st"]), first_diff=first,
                       floor_passes=sum(1 for p in d["log"] if p["floor"])))
        # I2: our CAI-DLLM rule with the authors' commit code as a probe
        probe = AuthorsCAI(cg, aps, pbs, lp + GEN, GEN // BLOCK_CAI)
        d = R.decode_batch(model, inp, None, args_cai, sched, ctrl, R.PassLog(0.9), cai_probe=probe)
        forced = {k: sum(p["cai"][k] for p in d["log"]) for k in ("n_budget", "n_grind", "n_floor")}
        i2.append(dict(problem=pid, passes=probe.passes, nfe=int(d["st"]), mismatched_passes=probe.mismatch,
                       token_mismatch_passes=probe.token_mismatch, first_mismatch=probe.first, forced_commits=forced,
                       pred=R.flexible_extract(tok.decode(d["out"][0, lp:], skip_special_tokens=True))))
        print(f"problem {pid}: I1 same tokens {i1[-1]['same_tokens']}, NFE authors {nfe_a} / port {i1[-1]['nfe_port']}; "
              f"I2 passes {probe.passes}, mismatched {probe.mismatch}, token mismatches {probe.token_mismatch}, "
              f"forced {forced}", flush=True)
    out = dict(n=len(ids), commits=commits, model_revision=a.model_revision,
               I1_pass=all(r["same_tokens"] and r["same_nfe"] for r in i1),
               I2_pass=all(r["mismatched_passes"] == 0 and r["passes"] == r["nfe"] for r in i2),
               I1=i1, I2=i2)
    print(f"I1 (DC-Leap port = authors' function): {out['I1_pass']}; "
          f"I2 (CAI-DLLM commit sets = authors' commit code at every pass): {out['I2_pass']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    sys.exit(0 if out["I1_pass"] and out["I2_pass"] else 1)


if __name__ == "__main__":
    main()
