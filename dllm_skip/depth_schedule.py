"""Depth schedules: which layers run at a given denoising state.

Ported from this project's prototype at commit 0b6905c (`dllm_skip/schedule.py`) with the
three defects `docs/01_experiment_plan.md` section 1 calls out or implies:

  * **`budget` is a per-bucket vector, not a curve.** The prototype only had the parametric
    form `k(r) = min + (max-min)(1-r)^gamma`, so schedule B -- the paper's actual method,
    whose per-bucket skip counts come out of the iso-error search and fit no gamma -- could
    not be represented by its own class. `budget` is now the primitive; `from_gamma` is one
    way to fill it and `from_iso_error` is the other.
  * **`no_consecutive` was declared but never implemented.** The prototype took `order[:k]`
    straight off the ranking. Non-adjacency is not cosmetic: with first/last protected it is
    what caps a whole-layer skip set at ceil((L-2)/2) -- 15 of 32, or 12 under keep_last=8 --
    and every ceiling in `06` section 2.5 Stage 0 is computed from it.
  * **`keep_last` defaulted to 1**, against the 8 of `01` section 1 (2603.07475 finds LLaDA's
    last third least redundant).

For Family B all three structural rules are hypotheses, not constraints (`01` section 1b,
D5): nothing is known about where an AR-converted block-diffusion model keeps its
redundancy, so a Stage 2 cell that violates them is a finding, not a disqualified cell.
Pass `keep_first=0, keep_last=0, no_consecutive=False` to test that.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class StepState:
    """What a controller may look at when choosing the depth of the next forward pass."""
    mask_ratio: float                        # masked fraction inside the current block
    block_idx: int = 0
    num_blocks: int = 1
    step_in_block: int = 0
    prev_conf_mean: Optional[float] = None   # mean max-prob over masked block tokens, prev pass
    prev_conf_min: Optional[float] = None    # min max-prob among tokens unmasked last pass
    prev_num_masked: Optional[int] = None
    is_cache_write: bool = False             # prefill / block-start / clean-block encode
    block_slice: Optional[Tuple[int, int]] = None   # (s, e) of the block in the canvas,
    # so a full-canvas pass can seed the `reuse` buffers by slicing to the block's rows
    # (`01` section 1b, 2026-09-09). None means the pass already covers only the block.


def bucket_of(mask_ratio: float, n_buckets: int) -> int:
    """Bucket 0 covers mask_ratio in (1-1/n, 1] (block start); bucket n-1 covers [0, 1/n]."""
    b = int(math.floor((1.0 - mask_ratio) * n_buckets))
    return min(max(b, 0), n_buckets - 1)


def _protected(n_layers: int, keep_first: int, keep_last: int) -> set:
    """The final layer is ALWAYS protected -- it feeds the LM head -- and `keep_last` extends
    that protection backwards. `scripts/analysis/stage0.py` encodes the same rule as
    `pool_free = L - 2` for "keep_last off", so keep_last=0 still leaves the last layer out;
    where the two disagree the script is right (`CLAUDE.md`)."""
    kl = max(int(keep_last), 1)
    return set(range(max(0, int(keep_first)))) | set(range(n_layers - kl, n_layers))


def max_skippable(n_layers: int, keep_first: int = 1, keep_last: int = 8,
                  no_consecutive: bool = True) -> int:
    """The structural ceiling `06` section 2.5 Stage 0 reports, recomputed here.

    Matches stage0.py: L=32 gives 12 at keep_last=8 and 15 with it off; L=28 gives 10 and 13;
    L=36 gives 14 and 17."""
    pool = max(0, n_layers - len(_protected(n_layers, keep_first, keep_last)))
    return -(-pool // 2) if no_consecutive else pool


def select_skips(order: Sequence[int], k: int, n_layers: int, keep_first: int = 1,
                 keep_last: int = 8, no_consecutive: bool = True) -> Tuple[int, ...]:
    """Take the k most-redundant layers off `order` subject to the structural rules.

    Greedy in ranking order: a candidate is taken unless it is protected or would sit next
    to one already taken. Greedy is not optimal in general, but it is what preserves the
    ranking's meaning -- the alternative (maximising the count) would reach the ceiling by
    picking {1,3,5,...}, which uses no information from the ranking at all and is exactly the
    degenerate top cell `06` section 2.5 Stage 0 warns about.

    **Greedy can block itself**, and on a real ranking it does. Family B's cosine order starts
    [11, 16, 12, 15, 2, ...]; taking 11 rules out 10 and 12, taking 16 rules out 15 and 17, and
    at k = 9 the ranking runs out after 7 admissible layers even though the structural ceiling
    is 10. Raising k is then not a structural impossibility but an artefact of the order the
    greedy consumed the ranking in. When and only when greedy comes up short, this falls back
    to `_max_weight_nonadjacent`, an exact DP that maximises total ranking weight over
    non-adjacent admissible sets of exactly k layers -- the same objective greedy approximates,
    solved rather than approximated. The fallback never fires when greedy succeeds, so every
    set selected before it existed (all of Family A) is unchanged and reproducible.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    ceiling = max_skippable(n_layers, keep_first, keep_last, no_consecutive)
    if k > ceiling:
        raise ValueError(
            f"cannot skip {k} of {n_layers} layers under keep_first={keep_first}, "
            f"keep_last={keep_last}, no_consecutive={no_consecutive}: ceiling is {ceiling}. "
            "`06` section 2.5 Stage 0 prescribes relaxing keep_last, and recording that the "
            "requirement reaches into the last third.")
    protected = _protected(n_layers, keep_first, keep_last)
    chosen: List[int] = []
    for l in order:
        if len(chosen) == k:
            break
        if l in protected or l in chosen:
            continue
        if no_consecutive and any(abs(l - c) == 1 for c in chosen):
            continue
        chosen.append(int(l))
    if len(chosen) < k:
        chosen = _max_weight_nonadjacent(order, k, n_layers, protected, no_consecutive)
    return tuple(sorted(chosen))


def _max_weight_nonadjacent(order, k, n_layers, protected, no_consecutive):
    """Exactly k admissible, pairwise non-adjacent layers maximising total ranking weight.

    Weight is the ranking position reversed, so layer `order[0]` is worth the most and a layer
    absent from the ranking is worth 0 -- the same preference greedy expresses, without
    greedy's inability to give back an early pick that blocks two better later ones. DP over
    layer index with state (index, chosen so far, previous index taken); L <= 40 and k <= 20,
    so it is microseconds.
    """
    rank = {int(l): i for i, l in enumerate(order)}
    n = len(order)
    cand = [l for l in range(n_layers) if l not in protected]
    w = {l: (n - rank[l]) if l in rank else 0 for l in cand}
    NEG = float("-inf")
    # best[(i, c, prev)] over cand[i:], c still to take, prev = cand index last taken or -1
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def best(i, c, prev):
        if c == 0:
            return 0.0, ()
        if i >= len(cand):
            return NEG, ()
        skip_v, skip_s = best(i + 1, c, prev)
        take_v, take_s = NEG, ()
        blocked = no_consecutive and prev >= 0 and abs(cand[i] - cand[prev]) == 1
        if not blocked:
            v, sub = best(i + 1, c - 1, i)
            if v != NEG:
                take_v, take_s = w[cand[i]] + v, (cand[i],) + sub
        return (take_v, take_s) if take_v >= skip_v else (skip_v, skip_s)

    val, sel = best(0, k, -1)
    best.cache_clear()
    if val == NEG:
        raise ValueError(
            f"cannot place {k} non-adjacent layers among {len(cand)} admissible ones")
    return list(sel)



def leq_share(mode, n_layers):
    """L_eq removed per skipped layer, by mode (`04` section 2a; `stage0.py` MODELS).

    `layer_steps` counts a skipped layer as zero whatever the mode, which is right for
    `identity` and `reuse` but wrong for the sub-layer modes: a `no-attn` layer still runs its
    feed-forward half, which is 67 % of a Family A layer's bytes and 87 % of a Family B
    layer's. Reporting raw layer counts as depth would have put `no-attn` at 0.10 when its
    real cost ratio is near 0.87.
    """
    u = {32: dict(attn=134.2, ffn=302.0, kv=16.8),
         28: dict(attn=58.7, ffn=407.4, kv=2.1)}[n_layers]
    total = u["attn"] + u["ffn"] + u["kv"]
    if mode in ("identity", "reuse"):
        return 1.0
    if mode == "no-ffn":
        return u["ffn"] / total
    if mode == "no-attn":
        return (u["attn"] + u["kv"]) / total
    raise ValueError(mode)


@dataclass
class DepthSchedule:
    """Static, mask-ratio-conditioned layer schedule.

    n_layers   : transformer block count.
    skip_order : one ranking per bucket, most-redundant first. A single flat list is shared
                 by every bucket (a ShortGPT-style global ranking).
    budget     : skip count per bucket. len(budget) == len(skip_order).
    """

    n_layers: int
    skip_order: List[List[int]]
    budget: List[int]
    keep_first: int = 1
    keep_last: int = 8
    no_consecutive: bool = True
    full_depth_on_cache_write: bool = True

    def __post_init__(self):
        if self.skip_order and isinstance(self.skip_order[0], int):
            self.skip_order = [list(self.skip_order)]
        if isinstance(self.budget, int):
            self.budget = [self.budget] * len(self.skip_order)
        if len(self.budget) != len(self.skip_order):
            if len(self.skip_order) == 1:
                self.skip_order = [list(self.skip_order[0]) for _ in self.budget]
            else:
                raise ValueError(
                    f"budget has {len(self.budget)} buckets, skip_order has "
                    f"{len(self.skip_order)}")
        self.budget = [int(b) for b in self.budget]
        # fail at construction, not mid-generation
        for b, (order, k) in enumerate(zip(self.skip_order, self.budget)):
            select_skips(order, k, self.n_layers, self.keep_first, self.keep_last,
                         self.no_consecutive)

    @property
    def n_buckets(self) -> int:
        return len(self.budget)

    def num_skip(self, mask_ratio: float) -> int:
        return self.budget[bucket_of(mask_ratio, self.n_buckets)]

    def active_layers(self, state: StepState) -> Tuple[int, ...]:
        if state.is_cache_write and self.full_depth_on_cache_write:
            return tuple(range(self.n_layers))
        b = bucket_of(state.mask_ratio, self.n_buckets)
        skipped = set(select_skips(self.skip_order[b], self.budget[b], self.n_layers,
                                   self.keep_first, self.keep_last, self.no_consecutive))
        return tuple(l for l in range(self.n_layers) if l not in skipped)

    # -- constructors --------------------------------------------------------------------
    @classmethod
    def from_gamma(cls, n_layers: int, skip_order, n_buckets: int = 4, max_skip: int = 8,
                   min_skip: int = 0, gamma: float = 2.0, **kw) -> "DepthSchedule":
        """Schedule A: the parametric budget curve, evaluated at each bucket's centre."""
        budget = []
        for b in range(n_buckets):
            r = 1.0 - (b + 0.5) / n_buckets           # bucket centre in mask-ratio units
            budget.append(int(round(min_skip + (max_skip - min_skip) * (1.0 - r) ** gamma)))
        return cls(n_layers=n_layers, skip_order=skip_order, budget=budget, **kw)

    @classmethod
    def from_iso_error(cls, n_layers: int, skip_order, budget, **kw) -> "DepthSchedule":
        """Schedule B: counts straight out of the iso-error search (`01` section 1,
        `calibrate_depth.greedy_iso_error_schedule`). No curve is fitted."""
        return cls(n_layers=n_layers, skip_order=skip_order, budget=list(budget), **kw)

    @classmethod
    def static(cls, n_layers: int, skip_order, k: int, n_buckets: int = 1, **kw):
        """The iso-rho control: one count for every state (`05` section 1, Phase 2)."""
        return cls(n_layers=n_layers, skip_order=skip_order, budget=[k] * n_buckets, **kw)

    # -- (de)serialisation ---------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "DepthSchedule":
        return cls(**json.loads(s))

    @classmethod
    def from_redundancy(cls, redundancy: Sequence[Sequence[float]], budget, **kw):
        """`redundancy[bucket][layer]`: higher = the layer changes its input less (e.g. the
        cosine between block input and output on masked positions, `01` section 1
        `profile_redundancy`)."""
        n_layers = len(redundancy[0])
        orders = [sorted(range(n_layers), key=lambda l, r=red: -r[l]) for red in redundancy]
        return cls(n_layers=n_layers, skip_order=orders, budget=list(budget), **kw)


@dataclass
class ConfidenceController:
    """Dynamic depth from the previous pass's confidence -- schedule C of `01` section 3.

    Handles the selection effect the static schedule cannot see: the tokens still masked late
    in a block are the ones the model was least sure about, so mask ratio alone understates
    how hard the remaining work is.
    """

    base: DepthSchedule
    thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])
    skips: List[int] = field(default_factory=lambda: [0, 4, 8, 12])
    hysteresis: int = 1                      # max increase in skip count per pass
    _last_skip: int = 0

    def active_layers(self, state: StepState) -> Tuple[int, ...]:
        if state.is_cache_write and self.base.full_depth_on_cache_write:
            self._last_skip = 0
            return tuple(range(self.base.n_layers))
        if state.prev_conf_mean is None:
            k = self.base.num_skip(state.mask_ratio)
        else:
            idx = sum(state.prev_conf_mean >= t for t in self.thresholds)
            k = self.skips[min(idx, len(self.skips) - 1)]
        k = min(k, self._last_skip + self.hysteresis,
                max_skippable(self.base.n_layers, self.base.keep_first, self.base.keep_last,
                              self.base.no_consecutive))
        self._last_skip = k
        b = bucket_of(state.mask_ratio, self.base.n_buckets)
        skipped = set(select_skips(self.base.skip_order[b], k, self.base.n_layers,
                                   self.base.keep_first, self.base.keep_last,
                                   self.base.no_consecutive))
        return tuple(l for l in range(self.base.n_layers) if l not in skipped)


@dataclass
class RegimeSchedule:
    """Depth that depends on *when* in the block the pass is -- Phase 0.4's time axis.

    `results/phase0.4/PREREG.md`: `early` skips only on passes whose block mask ratio r exceeds
    r*, `late` only on r <= r*, `static` on every refinement pass (the Phase 0.25 behaviour).
    r* is the masked fraction that splits refinement passes 50/50 by count, measured in Phase
    0.35 (`results/phase0.35/RESULTS.md`: 0.4688 on both families).

    This is a schedule, not a hook change: it reads `StepState.mask_ratio`, which the samplers
    already fill in, and delegates everything else to the base schedule -- so a regime cell and
    the static cell it is compared with differ in exactly one place.

    Cache-writing passes stay at full depth through the base's own guard (`01` section 0). The
    protected half is not "shallower by less"; it runs the full stack, which is why a regime's
    mean realised L_eq is about half the static cell's at the same k and why `06` section 2.57
    forbids comparing the two at equal k.
    """

    base: DepthSchedule
    regime: str = "static"                   # early | late | static
    r_star: float = 0.4688

    def __post_init__(self):
        if self.regime not in ("early", "late", "static"):
            raise ValueError(f"regime must be early|late|static, got {self.regime!r}")

    @property
    def n_layers(self) -> int:
        return self.base.n_layers

    @property
    def budget(self) -> List[int]:
        return self.base.budget

    def skips_this_pass(self, state: StepState) -> bool:
        if self.regime == "static":
            return True
        return (state.mask_ratio > self.r_star) if self.regime == "early" \
            else (state.mask_ratio <= self.r_star)

    def active_layers(self, state: StepState) -> Tuple[int, ...]:
        if state.is_cache_write and self.base.full_depth_on_cache_write:
            return tuple(range(self.base.n_layers))
        if not self.skips_this_pass(state):
            return tuple(range(self.base.n_layers))
        return self.base.active_layers(state)

    def to_json(self) -> str:
        return json.dumps(dict(regime=self.regime, r_star=self.r_star,
                               base=asdict(self.base)), indent=2)


def layer_steps(active_sets: Sequence[Sequence[int]]) -> int:
    """Executed (layer, pass) pairs. NOT an iso-cost axis across cache modes (`08` section 2);
    use measured wall-clock and token-weighted FLOPs for that."""
    return sum(len(s) for s in active_sets)


def _selftest():
    """The contract for `select_skips`, run as `python -m dllm_skip.depth_schedule`.

    Two things it pins. (1) The ceilings, against `scripts/analysis/stage0.py`. (2) That the
    DP fallback is *only* a fallback: every set Family A's Stage 2 cells ran on is reproduced
    exactly, because on Family A's ranking greedy never comes up short. If this ever fails,
    the sets in `results/phase0.25/RESULTS.md` no longer describe what ran.
    """
    assert max_skippable(32, 1, 8, True) == 12 and max_skippable(32, 1, 0, True) == 15
    assert max_skippable(28, 1, 8, True) == 10 and max_skippable(28, 1, 0, True) == 13
    assert max_skippable(36, 1, 8, True) == 14 and max_skippable(36, 1, 0, True) == 17

    # Family A, global cosine order of results/phase0/calib_cosine.json (first 16 entries are
    # all the selection ever reaches at k <= 14)
    A = [2, 5, 7, 9, 11, 14, 16, 18, 20, 22, 1, 3, 4, 6, 8, 10, 12, 13, 15, 17, 19, 21,
         23, 24, 25, 26, 27, 28, 29, 30, 0, 31]
    assert select_skips(A, 4, 32, 1, 8, True) == (2, 5, 7, 9)
    assert select_skips(A, 7, 32, 1, 8, True) == (2, 5, 7, 9, 11, 14, 16)
    assert select_skips(A, 10, 32, 1, 8, True) == (2, 5, 7, 9, 11, 14, 16, 18, 20, 22)

    # Family B, the ranking that made greedy block itself: [11, 16, ...] takes 11 (ruling out
    # 10, 12) and 16 (ruling out 15, 17), and at k = 9 greedy dies at 7 of 9 with the ceiling
    # at 10. The fallback must return exactly 9 admissible, non-adjacent layers.
    B = [11, 16, 12, 15, 2, 17, 14, 5, 13, 10, 8, 4, 7, 6, 9, 18, 3, 19,
         24, 23, 20, 25, 22, 21, 26, 1, 27, 0]
    assert select_skips(B, 3, 28, 1, 8, True) == (2, 11, 16)
    assert select_skips(B, 6, 28, 1, 8, True) == (2, 5, 8, 11, 14, 16)
    # RegimeSchedule: the protected half must run the full stack, and a cache-writing pass must
    # run it in every regime. Getting this wrong is silent -- the cell just becomes another
    # static cell -- so it is pinned here rather than checked by eye in a log.
    base = DepthSchedule.static(32, [A], 7, keep_first=1, keep_last=8, no_consecutive=True)
    for regime, hi_full, lo_full in (("early", False, True), ("late", True, False),
                                     ("static", False, False)):
        rs = RegimeSchedule(base=base, regime=regime, r_star=0.4688)
        hi = rs.active_layers(StepState(mask_ratio=0.90, is_cache_write=False))
        lo = rs.active_layers(StepState(mask_ratio=0.10, is_cache_write=False))
        cw = rs.active_layers(StepState(mask_ratio=1.00, is_cache_write=True))
        assert (len(hi) == 32) == hi_full, (regime, len(hi))
        assert (len(lo) == 32) == lo_full, (regime, len(lo))
        assert len(cw) == 32, (regime, len(cw))
        assert len(hi) in (25, 32) and len(lo) in (25, 32), (regime, len(hi), len(lo))
    # r* is a strict upper bound on the `late` half: a pass exactly at r* is late, not early
    rs = RegimeSchedule(base=base, regime="late", r_star=0.4688)
    assert len(rs.active_layers(StepState(mask_ratio=0.4688, is_cache_write=False))) == 25

    s9 = select_skips(B, 9, 28, 1, 8, True)
    assert len(s9) == 9 and len(set(s9)) == 9, s9
    assert all(0 < l < 20 for l in s9), s9                       # admissible: keep_first/last
    assert all(b - a > 1 for a, b in zip(s9, s9[1:])), s9        # non-adjacent
    print("select_skips selftest: ok")


if __name__ == "__main__":
    _selftest()
