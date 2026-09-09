"""Contract for the PREREG §7 switches — `python check_diag_switches.py`, no GPU needed.

This file exists because the first D2 run silently measured the wrong thing. `skip_cache_writes`
cleared `_Depth`'s own `force_full` and stopped there, and the cell reported layer_steps
identical to the unmodified prefix k = 6 cell of Phase 0 step 4. The full-depth cache-write rule
of `01` §0 is enforced in **three** independent places, and a diagnostic that means to break it
has to lift all three:

  1. `SkipController.allow_cache_write_skip` -- the hook raises `SkipError` otherwise
  2. `DepthSchedule.full_depth_on_cache_write` -- `active_layers` returns every layer first,
     before the controller is ever armed. **This is the one that was missed**, and no test
     using a stub schedule can see it, because a stub is exactly the object that does not have
     the guard.
  3. the caller's `force_full=True` argument, cleared per pass in `_Depth.arm`

So this checks against the *real* `DepthSchedule` and the *real* calibration file, and it
models the two samplers' pass numbering separately, which also differs and matters:

  * `generate` (no cache) has no cache-writing pass at all and numbers its refinement passes
    from 0, so D1's "first pass of the block" is `step_in_block == 0`.
  * `generate_with_prefix_cache` numbers the block-start cache-writing pass 0 and its
    refinement passes from 1.

The `expected layer_steps` lines are the arithmetic the job logs are read against: with the
switch live D2 is 256*26 per problem, without it 8*32 + 248*26, and those differ by 96 -- small
enough that only an exact expectation catches it.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

CALIB = "/home1/hyunhoko/DLLM/results/phase0/calib_cosine.json"


def _load_depth():
    """Import `_Depth` without importing torch: the class itself has no tensor dependency."""
    ds = {}
    exec(compile(open(os.path.join(ROOT, "dllm_skip", "depth_schedule.py")).read(),
                 "depth_schedule", "exec"), ds)
    src = open(os.path.join(HERE, "generate.py")).read()
    i, j = src.index("class _Depth"), src.index("def _n_layers")
    ns = {"StepState": ds["StepState"]}
    exec(compile(src[i:j], "generate", "exec"), ns)
    return ds["DepthSchedule"], ns["_Depth"], ns["_state"]


class _Ctrl:
    n_layers = 32
    mode = "identity"
    store_deltas = False
    block_slice = None
    reuse_m = None
    allow_cache_write_skip = False

    def arm(self, act):
        self.last = act


def main():
    DepthSchedule, _Depth, _state = _load_depth()
    order = json.load(open(CALIB))["global_order"]

    def depths(path, **kw):
        sch = DepthSchedule.static(32, [order], 6, keep_first=1, keep_last=8,
                                   no_consecutive=True)
        c = _Ctrl()
        d = _Depth(sch, c, 32, log=[], **kw)
        if path == "prefix":                       # block-start pass is step 0, refine from 1
            d.arm(_state(1.00, 0, 8, 0, True), force_full=True)
            d.arm(_state(0.97, 0, 8, 1, False))
            d.arm(_state(0.94, 0, 8, 2, False))
        else:                                      # no cache: no cache-writing pass at all
            d.arm(_state(1.00, 0, 8, 0, False))
            d.arm(_state(0.97, 0, 8, 1, False))
            d.arm(_state(0.94, 0, 8, 2, False))
        return [r["depth"] for r in d.log], c.allow_cache_write_skip, sch.full_depth_on_cache_write

    cases = [
        ("prefix, no switch", "prefix", {}, [32, 26, 26], False, True),
        ("none,   no switch", "none", {}, [26, 26, 26], False, True),
        ("D1  none + protect_first_pass", "none",
         dict(protect_first_pass=True), [32, 26, 26], False, True),
        ("D2  prefix + skip_cache_writes", "prefix",
         dict(skip_cache_writes=True), [26, 26, 26], True, False),
    ]
    bad = 0
    for name, path, kw, want, want_hook, want_sched in cases:
        got, hook, sched = depths(path, **kw)
        ok = got == want and hook == want_hook and sched == want_sched
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {name:32s} depths={got} "
              f"hook_guard_lifted={hook} sched_guard={sched}")

    # the numbers the job log is read against, n = 100, gen 256, block 32 -> 8 blocks x 32 passes
    per = 256
    print()
    print(f"  expected layer_steps, n=100, k=6 (26 of 32 active):")
    print(f"    prefix, no switch / D1 none+protect : {100*(8*32 + (per-8)*26):>8d}   "
          f"(800 passes held at full depth)")
    print(f"    none, no switch                     : {100*per*26:>8d}")
    print(f"    D2 prefix + skip_cache_writes       : {100*per*26:>8d}   "
          f"(nothing held at full depth)")
    if bad:
        raise SystemExit(f"{bad} case(s) wrong")
    print("\n  diag switch contract: ok")


if __name__ == "__main__":
    main()
