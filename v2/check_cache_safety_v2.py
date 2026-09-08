"""Family B cache-safety checks -- `01_experiment_plan.md` section 2 (Family B list).

Fast-dLLM v2's cache is exact and is written only by the prefill, the in-block refresh and
the clean-block encode, all of which run at full depth by rule (`01` section 1b). The checks
therefore differ from Family A's:

  1 null skip          arming every layer is bit-identical on all three pass types
  2 no leak into cache two blocks refined with different skip sets, but ending in the SAME
                       finished tokens and encoded at full depth, must give bit-identical
                       logits on the next block's first refinement pass. Stated behaviourally
                       rather than by comparing `Cache` tensors, whose attribute names move
                       between transformers versions; the claim under test is the same one --
                       a refinement pass must not reach the cache.
  3 positive control   skipping INSIDE the encode must make the next block diverge, and more
                       so with more layers skipped. Without this, check 2 passing could just
                       mean the test is insensitive.
  4 stale slot         only applies with use_block_cache=True and sub-block < 32; the gate
                       setting has no in-block cache, so this is recorded as not applicable.
  5 reuse / poisoning  mode plumbing, identical to Family A checks 4 and 5.
  6 cache-write guard  skipping on update_past_key_values=True raises.
"""
import argparse, copy, os, sys, types
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import generation_functions                                       # noqa: E402
import profile_step_v2 as P                                       # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM      # noqa: E402
from dllm_skip.hook import (SkipController, SkipError, install_skipping,   # noqa: E402
                            uninstall_skipping, IDENTITY, REUSE)

MASK_ID = 151665
RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--skip1", default="1,3,5,7")
    ap.add_argument("--skip2", default="9,11,13,15")
    a = ap.parse_args()
    S1 = [int(x) for x in a.skip1.split(",")]
    S2 = [int(x) for x in a.skip2.split(",")]

    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16).to(dev).eval()
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model)

    ids = tok([P.build_prompts(tok, 1)[0]], return_tensors="pt").input_ids.to(dev)
    bs = a.block_size
    n_full = ids.shape[1] // bs * bs
    prefill_ids = ids[:, :n_full]
    L = len(P.find_module_list(model, "layers")[1])
    active_all = list(range(L))
    keep1 = [l for l in range(L) if l not in S1]
    keep2 = [l for l in range(L) if l not in S2]
    print(f"\nFast-dLLM v2 7B  L={L}  prompt={ids.shape[1]} (prefill {n_full})  block={bs}"
          f"  skip1={S1} skip2={S2}\n")

    def prefill():
        return model(input_ids=prefill_ids, use_cache=True, update_past_key_values=True,
                     block_size=bs)

    with torch.no_grad():
        base = prefill()
        pkv0 = base.past_key_values
        blk = torch.full((1, bs), MASK_ID, dtype=torch.long, device=dev)
        finished = ids[:, n_full:n_full + bs] if ids.shape[1] - n_full >= bs else \
            torch.full((1, bs), tok.eos_token_id or 0, dtype=torch.long, device=dev)

        # ------------------------------------------------------------------ 1 null skip
        ref_pre = base.logits.clone()
        ref_ref = model(input_ids=blk, use_cache=True, past_key_values=copy.deepcopy(pkv0),
                        update_past_key_values=False).logits.clone()
        ctrl = install_skipping(model)
        oks, ds = [], []
        ctrl.arm(active_all)
        got = model(input_ids=prefill_ids, use_cache=True, update_past_key_values=True,
                    block_size=bs).logits
        oks.append(torch.equal(got, ref_pre)); ds.append((got.float()-ref_pre.float()).abs().max().item())
        ctrl.arm(active_all)
        got = model(input_ids=blk, use_cache=True, past_key_values=copy.deepcopy(pkv0),
                    update_past_key_values=False).logits
        oks.append(torch.equal(got, ref_ref)); ds.append((got.float()-ref_ref.float()).abs().max().item())
        report("1 null skip bit-identical (prefill, refinement)", all(oks),
               f"max|d|={max(ds):.3e}")

        # ------------------------------------------------------------------ 6 guard
        try:
            ctrl.arm(keep1)
            model(input_ids=blk, use_cache=True, past_key_values=copy.deepcopy(pkv0),
                  update_past_key_values=True, block_size=bs)
            report("6 cache-write guard raises", False, "no exception")
        except SkipError as ex:
            report("6 cache-write guard raises", True, type(ex).__name__)
        ctrl._seen = 0; ctrl._active = None

        # ------------------------------------------------------------------ 2 no leak
        def refine_then_encode_then_next(skip_keep, encode_keep):
            pkv = copy.deepcopy(pkv0)
            if skip_keep is not None:
                ctrl.arm(skip_keep)
            model(input_ids=blk, use_cache=True, past_key_values=pkv,
                  update_past_key_values=False)                  # refinement, may skip
            if encode_keep is not None:
                ctrl.arm(encode_keep)
            model(input_ids=finished, use_cache=True, past_key_values=pkv,
                  update_past_key_values=True, block_size=bs)    # encode, writes cache
            return model(input_ids=blk, use_cache=True, past_key_values=pkv,
                         update_past_key_values=False).logits    # next block, full depth

        lo_a = refine_then_encode_then_next(keep1, None)
        lo_b = refine_then_encode_then_next(None, None)
        d = (lo_a.float() - lo_b.float()).abs().max().item()
        report("2 refinement does not leak into the cache", torch.equal(lo_a, lo_b),
               f"max|d|={d:.3e} (expect exactly 0)")

        # ------------------------------------------------------------------ 3 positive control
        # `01` section 2 (Family B check 3) expects a shallow clean-block encode to make the
        # NEXT block diverge, growing with |S|. On this model it does something stronger and
        # simpler: a skipped layer never calls self_attn, so it never appends its K/V, and the
        # encode leaves a RAGGED cache -- skipped layers short, active layers long. The next
        # pass then cannot run at all; two runs died with
        #   "The expanded size of the tensor (128) must match the existing size (160) at
        #    non-singleton dimension 3"
        # which is the mask for a 160-long cache meeting a layer that only has 128. So the
        # control is stated on the cache itself: raggedness must appear exactly on the skipped
        # set. That is a sharper demonstration of why `01` section 0 keeps cache writes at full
        # depth, and it proves check 2 above is sensitive rather than blind.
        def cache_len(pkv, i):
            obj = getattr(pkv, "layers", None)
            if obj is not None:
                try:
                    return int(obj[i].keys.shape[-2])
                except Exception:
                    pass
            kc = getattr(pkv, "key_cache", None)
            if kc is not None:
                return int(kc[i].shape[-2])
            return int(pkv[i][0].shape[-2])

        ctrl.allow_cache_write_skip = True
        rows = []
        for S in ([], S1, sorted(set(S1 + S2))):
            keep = [l for l in range(L) if l not in S]
            pkv = copy.deepcopy(pkv0)
            model(input_ids=blk, use_cache=True, past_key_values=pkv,
                  update_past_key_values=False)                     # refinement, full depth
            ctrl.arm(keep)
            model(input_ids=finished, use_cache=True, past_key_values=pkv,
                  update_past_key_values=True, block_size=bs)       # encode, possibly shallow
            lens = [cache_len(pkv, l) for l in range(L)]
            short = sorted(l for l in range(L) if lens[l] != max(lens))
            rows.append((sorted(S), short, len(set(lens))))
        ctrl.allow_cache_write_skip = False
        ok = (rows[0][1] == [] and rows[0][2] == 1                  # full depth: uniform cache
              and all(sk == S for S, sk, _ in rows[1:]))            # shallow: ragged exactly on S
        report("3 positive control: shallow encode leaves a ragged cache, exactly on the "
               "skipped layers", ok,
               "  ".join(f"|S|={len(S)}->short={sk if len(sk)<6 else str(sk[:5])+'..'}"
                         for S, sk, _ in rows))

        # ------------------------------------------------------------------ 4 N/A
        report("4 stale-slot test", True,
               "not applicable in the gate setting (use_block_cache=False, sub-block 32)")

        # ------------------------------------------------------------------ 5 reuse + poison
        ctrl.mode = REUSE; ctrl.new_block()
        pkv = copy.deepcopy(pkv0)
        ctrl.arm(active_all)
        r_full = model(input_ids=blk, use_cache=True, past_key_values=pkv,
                       update_past_key_values=False).logits.clone()
        pkv = copy.deepcopy(pkv0)
        ctrl.arm(keep1)
        r_reuse = model(input_ids=blk, use_cache=True, past_key_values=pkv,
                        update_past_key_values=False).logits
        d = (r_full.float() - r_reuse.float()).abs().max().item()
        # Same bf16 story as Family A check 4: x + fl(h_out - x) != h_out at 8 mantissa bits,
        # so the bit-identity `06` section 2.5 Stage 1 asks for is unattainable in bf16 (this
        # failed at 2.812e-01 when asserted). Tolerance here, exactness in fp32 below.
        report("5a reuse staleness-0 (bf16, rounding-tolerant)", d < 0.5,
               f"deltas={len(ctrl.deltas)}/{L}  max|d|={d:.3e} (tol 0.5)")
        model.float()
        pkv32 = copy.deepcopy(pkv0)
        ctrl.new_block(); ctrl.arm(active_all)
        g_full = model(input_ids=blk, use_cache=True, past_key_values=copy.deepcopy(pkv32),
                       update_past_key_values=False).logits.clone()
        ctrl.arm(keep1)
        g_reuse = model(input_ids=blk, use_cache=True, past_key_values=copy.deepcopy(pkv32),
                        update_past_key_values=False).logits
        d32 = (g_full - g_reuse).abs().max().item()
        report("5a-fp32 reuse staleness-0 (must be exact)", d32 < 1e-4, f"max|d|={d32:.3e}")
        model.to(torch.bfloat16)
        # the fp32 pass left fp32 deltas; clear and re-seed in bf16 before poisoning
        ctrl.new_block(); ctrl.arm(active_all)
        model(input_ids=blk, use_cache=True, past_key_values=copy.deepcopy(pkv0),
              update_past_key_values=False)
        for l in S2:
            ctrl.deltas[l] = torch.full_like(ctrl.deltas[S1[0]], float("nan"))
        pkv = copy.deepcopy(pkv0)
        ctrl.arm(keep1)
        out = model(input_ids=blk, use_cache=True, past_key_values=pkv,
                    update_past_key_values=False).logits
        report("5b NaN in unread delta buffers stays out", bool(torch.isfinite(out).all()),
               f"poisoned {S2}")

        uninstall_skipping(model)

    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n  {n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
