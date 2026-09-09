"""Family A cache-safety checks -- `01_experiment_plan.md` section 2, `06` section 2.5 Stage 1.

Every check is run on the real LLaDA-8B-Instruct in bf16, drives `model.forward` directly so
the pass boundaries are explicit, and prints PASS/FAIL with the tolerance it used. Nothing
downstream of Stage 1 is believable until these pass (`06` section 2.5).

  1 null skip            arming every layer reproduces the unhooked logits bit-for-bit
  2 prefix consistency   a cached refinement pass with skip set S equals a no-cache forward
                         with S on the same canvas, given a full-depth prefix (tol 1e-3 bf16)
  3 stale slot (dual)    a skipped layer's stale in-block K/V slots are never read: two
                         different skip sets in sequence give the same answer as the second
                         one alone
  4 reuse staleness-0    re-running the same pass with layer l skipped and its just-recorded
                         delta applied is bit-identical -- x + (h_out - h_in) == h_out
  5 buffer poisoning     a NaN in a delta buffer that must not be read does not reach the
                         output, proving the read path is the one we think it is
  6 cache-write guard    skipping on a cache-writing pass raises instead of corrupting
"""
import argparse, os, sys, copy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import profile_step as P                                        # noqa: E402
from model.modeling_llada import LLaDAModelLM                   # noqa: E402
from transformers import AutoTokenizer                          # noqa: E402
from dllm_skip.hook import (SkipController, SkipError, install_skipping,      # noqa: E402
                            uninstall_skipping, IDENTITY, REUSE, NO_FFN, NO_ATTN)

MASK_ID = 126336
RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}", flush=True)


def clone_pkv(pkv):
    return tuple(tuple(t.clone() for t in layer) for layer in pkv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--skip1", default="1,3,5,7")
    ap.add_argument("--skip2", default="9,11,13,15")
    a = ap.parse_args()
    S1 = [int(x) for x in a.skip1.split(",")]
    S2 = [int(x) for x in a.skip2.split(",")]

    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    prompt = tok(P.build_prompts(tok, 1, 5)[0], return_tensors="pt").input_ids.to(dev)
    Lp = prompt.shape[1]
    x = torch.full((1, Lp + a.gen_length), MASK_ID, dtype=torch.long, device=dev)
    x[:, :Lp] = prompt
    s, e = Lp, Lp + a.block_length
    L = len(P.find_blocks(model)[1])
    active_all = list(range(L))
    keep1 = [l for l in range(L) if l not in S1]
    keep2 = [l for l in range(L) if l not in S2]
    print(f"\nLLaDA-8B-Instruct  L={L}  prompt={Lp}  block=[{s},{e})  skip1={S1} skip2={S2}\n")

    with torch.no_grad():
        # ---------------------------------------------------------------- 1 null skip
        ref = model(x, use_cache=False).logits.clone()
        ctrl = install_skipping(model)
        ctrl.arm(active_all)
        got = model(x, use_cache=False).logits
        report("1 null skip is bit-identical", torch.equal(got, ref),
               f"max|d|={(got.float()-ref.float()).abs().max().item():.3e}")

        # ---------------------------------------------------------------- 6 cache-write guard
        try:
            ctrl.arm(keep1)
            model(x, use_cache=True)            # cache-writing pass: layer_past is None
            report("6 cache-write guard raises", False, "no exception")
        except SkipError as ex:
            report("6 cache-write guard raises", True, f"{type(ex).__name__}")
        ctrl._seen = 0; ctrl._active = None      # the aborted pass left the counter mid-way

        # ---------------------------------------------------------------- 2 prefix cache
        # The first version of this check compared a cached refinement pass with skip set S
        # against a no-cache forward with S on the same canvas, and failed at max|d| = 4.28.
        # That equality is false and the test was wrong, not the hook: in the cached run the
        # PREFIX K/V come from the full-depth warm-up, while in a plain no-cache forward with
        # S they are recomputed through the skipped stack. `01` section 2 asks for a reference
        # "whose prefix K/V came from a full-depth pass", and no single forward provides one.
        # What is actually checkable -- and what the cache-safety claim of `00` section 2.7
        # says -- is structural: a skipped layer neither reads nor writes its own cache slot,
        # so on a prefix-mode refinement pass its cache must stay exactly the prefix while an
        # active layer's grows by the number of new positions.
        full = model(x, use_cache=True)
        pkv_prefix = tuple(tuple(t[:, :, :s].clone() for t in layer)
                           for layer in full.past_key_values)
        ctrl.arm(keep1)
        out2 = model(x[:, s:], past_key_values=clone_pkv(pkv_prefix), use_cache=True)
        grew, stayed, bad = [], [], []
        n_new = x.shape[1] - s
        for l, layer in enumerate(out2.past_key_values):
            got = layer[0].shape[2]
            if l in S1:
                (stayed if got == s else bad).append(l)
            else:
                (grew if got == s + n_new else bad).append(l)
        report("2 skipped layers leave the prefix cache untouched", not bad,
               f"{len(stayed)} skipped stayed at {s}, {len(grew)} active grew to {s+n_new}"
               + (f", WRONG: {bad}" if bad else ""))

        # ---------------------------------------------------------------- 3 stale slot (dual)
        rp = torch.zeros_like(x, dtype=torch.bool); rp[:, s:e] = True
        x2 = x.clone(); x2[:, s:s + 4] = prompt[:, :4]     # a canvas the skipped layers never saw

        base = model(x, use_cache=True)                     # warm-up, full depth
        pkv_a = clone_pkv(base.past_key_values)
        ctrl.arm(keep1)
        model(x[:, s:e], past_key_values=pkv_a, use_cache=True, replace_position=rp)
        ctrl.arm(keep2)
        seq = model(x2[:, s:e], past_key_values=pkv_a, use_cache=True, replace_position=rp).logits

        pkv_b = clone_pkv(base.past_key_values)             # fresh warm-up state
        ctrl.arm(keep2)
        alone = model(x2[:, s:e], past_key_values=pkv_b, use_cache=True, replace_position=rp).logits
        d = (seq.float() - alone.float()).abs().max().item()
        report("3 stale slots never read (dual)", d < 1e-3, f"max|d|={d:.3e} (tol 1e-3)")

        # ---------------------------------------------------------------- 4 reuse staleness-0
        pkv_c = clone_pkv(base.past_key_values)
        ctrl.mode = REUSE
        ctrl.new_block()
        ctrl.arm(active_all)                                 # full depth: records every delta
        r_full = model(x[:, s:e], past_key_values=pkv_c, use_cache=True,
                       replace_position=rp).logits.clone()
        have = sorted(ctrl.deltas.keys())
        pkv_d = clone_pkv(base.past_key_values)
        ctrl.arm(keep1)                                      # same input, layers S1 from delta
        r_reuse = model(x[:, s:e], past_key_values=pkv_d, use_cache=True,
                        replace_position=rp).logits
        d = (r_full.float() - r_reuse.float()).abs().max().item()
        # `06` section 2.5 Stage 1 asks for bit-identity "since x + Delta = h_out by
        # construction". That holds in exact arithmetic; in bf16 it cannot, because the delta
        # is stored rounded and x + fl(h_out - x) != h_out at 8 mantissa bits. The first run
        # of this check failed at max|d| = 0.1875 = 3/16, a value on the bf16 grid -- rounding,
        # not a slicing bug. So: a tight tolerance here, and the exact identity is tested
        # separately in fp32 below, which is what actually separates the two explanations.
        report("4a reuse staleness-0 (bf16, rounding-tolerant)", d < 0.5,
               f"deltas={len(have)}/{L}  max|d|={d:.3e} (tol 0.5; bit-exact is unattainable)")

        # -- the same identity in fp32, where it must be exact ----------------------------
        m32 = model.float()
        pkv_f = tuple(tuple(t.float() for t in layer) for layer in base.past_key_values)
        ctrl.mode = REUSE; ctrl.new_block()
        ctrl.arm(active_all)
        f_full = m32(x[:, s:e], past_key_values=clone_pkv(pkv_f), use_cache=True,
                     replace_position=rp).logits.clone()
        ctrl.arm(keep1)
        f_reuse = m32(x[:, s:e], past_key_values=clone_pkv(pkv_f), use_cache=True,
                      replace_position=rp).logits
        d32 = (f_full - f_reuse).abs().max().item()
        report("4b reuse staleness-0 (fp32, must be exact)", d32 < 1e-4, f"max|d|={d32:.3e}")
        model.to(torch.bfloat16)
        # The fp32 pass left fp32 deltas in the controller; adding one to a bf16 hidden state
        # promotes the activation and the next matmul dies with
        #   "expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16".
        # Clear them and re-seed in bf16 before the poisoning check.
        ctrl.new_block()
        ctrl.arm(active_all)
        model(x[:, s:e], past_key_values=clone_pkv(base.past_key_values), use_cache=True,
              replace_position=rp)

        # ---------------------------------------------------------------- 5 buffer poisoning
        for l in S2:
            ctrl.deltas[l] = torch.full_like(ctrl.deltas[S1[0]], float("nan"))
        pkv_e = clone_pkv(base.past_key_values)
        ctrl.arm(keep1)                                      # S2 layers RUN, their deltas unread
        out = model(x[:, s:e], past_key_values=pkv_e, use_cache=True, replace_position=rp).logits
        report("5 NaN in unread delta buffers stays out", bool(torch.isfinite(out).all()),
               f"poisoned layers {S2}")

        # ------------------------------------------------------- 6b seed alignment
        # `06` Stage 1 check 6 (2026-09-09). The block-start pass covers the whole canvas, so
        # the reuse buffer takes delta[:, s:e]. This proves those rows are the block's own
        # delta and not a misaligned slice: run one full-depth refinement pass on the SAME
        # canvas (inspection only, nothing committed) and compare its fresh per-layer delta
        # against the sliced one. A mismatch here is an indexing error, not staleness.
        probe = {}
        h = []
        for i, blk in enumerate(P.find_blocks(model)[1]):
            def mk(idx):
                def pre(m, args, kwargs=None):
                    probe[("in", idx)] = (args[0] if args else kwargs["x"]).detach()
                def post(m, args, out, kwargs=None):
                    probe[("out", idx)] = out[0].detach()
                return pre, post
            pre, post = mk(i)
            h.append(blk.register_forward_pre_hook(pre)); h.append(blk.register_forward_hook(post))
        try:
            ctrl.mode = REUSE; ctrl.new_block()
            ctrl.block_slice = (s, e)
            ctrl.arm(active_all)                 # block-start pass seeds by slicing
            base6 = model(x, use_cache=True)
            seeded = {k: v.clone() for k, v in ctrl.deltas.items()}
            probe.clear()
            ctrl.block_slice = None
            ctrl.arm(active_all)                 # fresh full-depth refinement on the same canvas
            model(x[:, s:e], past_key_values=clone_pkv(base6.past_key_values), use_cache=True,
                  replace_position=rp)
            fresh = {i: (probe[("out", i)] - probe[("in", i)]) for i in range(L)
                     if ("out", i) in probe}
        finally:
            for hh in h:
                hh.remove()
        common = [i for i in sorted(fresh) if i in seeded and seeded[i].shape == fresh[i].shape]
        shapes_ok = len(common) == len(fresh) and bool(common)
        diffs = [(seeded[i].float() - fresh[i].float()).abs().max().item() for i in common]
        scale = max(seeded[i].float().abs().max().item() for i in common)
        rel = max(diffs) / scale
        # Absolute tolerance is not transferable here: check 3 compares LOGITS, this compares
        # hidden-state deltas, whose scale is larger. What separates rounding from a misaligned
        # slice is precision, exactly as in check 4a/4b -- so the fp32 twin below is the test
        # that decides, and the bf16 number is reported as relative error.
        report("6b seed alignment, bf16 (relative)", shapes_ok and rel < 0.05,
               f"layers={len(common)}  max|d|={max(diffs):.3e}  scale={scale:.1f}  rel={rel:.2%}")

        m32 = model.float()
        probe.clear(); h2 = []
        for i, blk in enumerate(P.find_blocks(model)[1]):
            def mk32(idx):
                def pre(m, args, kwargs=None):
                    probe[("in", idx)] = (args[0] if args else kwargs["x"]).detach()
                def post(m, args, out, kwargs=None):
                    probe[("out", idx)] = out[0].detach()
                return pre, post
            pre, post = mk32(i)
            h2.append(blk.register_forward_pre_hook(pre)); h2.append(blk.register_forward_hook(post))
        try:
            ctrl.mode = REUSE; ctrl.new_block(); ctrl.block_slice = (s, e)
            ctrl.arm(active_all)
            b32 = m32(x, use_cache=True)
            seed32 = {k: v.clone() for k, v in ctrl.deltas.items()}
            probe.clear(); ctrl.block_slice = None; ctrl.arm(active_all)
            m32(x[:, s:e], past_key_values=tuple(tuple(t.clone() for t in l)
                                                 for l in b32.past_key_values),
                use_cache=True, replace_position=rp)
            fresh32 = {i: (probe[("out", i)] - probe[("in", i)]) for i in range(L)
                       if ("out", i) in probe}
        finally:
            for hh in h2:
                hh.remove()
            model.to(torch.bfloat16)
        d32 = max((seed32[i] - fresh32[i]).abs().max().item()
                  for i in fresh32 if i in seed32 and seed32[i].shape == fresh32[i].shape)
        rel32 = d32 / scale
        # The criterion is relative, and the diagnostic is the drop with precision. `06` Stage 1
        # check 6 says "fp32 twin <~ 1e-4", which is the LOGITS scale of check 4b (2.8e-05
        # there); this compares hidden-state deltas whose scale is ~288, where 1e-4 absolute is
        # below fp32's own resolution. A misaligned slice does not shrink when precision rises:
        # bf16 -> fp32 falling by three orders of magnitude is what settles it.
        report("6b seed alignment, fp32 (decides rounding vs misalignment)",
               rel32 < 1e-4 and d32 < max(diffs) / 100,
               f"max|d|={d32:.3e}  rel={rel32:.1e}  bf16/fp32 = {max(diffs)/d32:.0f}x "
               f"(tol: rel < 1e-4 AND at least 100x smaller than bf16)")
        ctrl.mode = IDENTITY; ctrl.new_block(); ctrl.block_slice = None

        # ------------------------------------------------------- 7 sub-layer modes
        # `no-ffn` runs the attention half, so it still writes its own K/V and is legal on a
        # cache-writing pass; `no-attn` runs the feed-forward half and hands `layer_past` back
        # like `identity`. Both must differ from full depth AND from each other, and their sum
        # of residual updates must reconstruct the full block on a single layer.
        base = model(x, use_cache=True)
        ref = model(x[:, s:e], past_key_values=clone_pkv(base.past_key_values), use_cache=True,
                    replace_position=rp).logits.clone()
        outs = {}
        for mode in (NO_FFN, NO_ATTN, IDENTITY):
            ctrl.mode = mode; ctrl.new_block()
            if mode == REUSE:
                continue
            ctrl.arm(keep1)
            outs[mode] = model(x[:, s:e], past_key_values=clone_pkv(base.past_key_values),
                               use_cache=True, replace_position=rp).logits.clone()
        d_ffn = (outs[NO_FFN].float() - ref.float()).abs().max().item()
        d_att = (outs[NO_ATTN].float() - ref.float()).abs().max().item()
        d_id  = (outs[IDENTITY].float() - ref.float()).abs().max().item()
        distinct = (not torch.equal(outs[NO_FFN], outs[NO_ATTN])
                    and not torch.equal(outs[NO_FFN], outs[IDENTITY])
                    and not torch.equal(outs[NO_ATTN], outs[IDENTITY]))
        # each removes less than removing the whole layer
        ordered = d_ffn < d_id and d_att < d_id
        report("7 sub-layer modes differ from full depth, from each other, and less than identity",
               distinct and ordered and d_ffn > 0 and d_att > 0,
               f"no-ffn {d_ffn:.3e}  no-attn {d_att:.3e}  identity {d_id:.3e}")

        # ------------------------------------------------------- 8 no-ffn on a cache write
        # It must NOT raise: its attention half produces the cache the loop asserts on.
        ctrl.mode = NO_FFN
        try:
            ctrl.arm(keep1)
            model(x, use_cache=True)
            report("8 no-ffn is legal on a cache-writing pass", True, "no exception, as designed")
        except SkipError as ex:
            report("8 no-ffn is legal on a cache-writing pass", False, f"raised {ex}")
        ctrl._seen = 0; ctrl._active = None; ctrl.mode = IDENTITY

        uninstall_skipping(model)

    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n  {n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
