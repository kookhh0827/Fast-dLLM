"""Layer skipping for both model families, without editing either model file.

Implements the semantics `docs/01_experiment_plan.md` section 1 specifies -- a per-pass
active-layer set, identity or cached-delta pass-through for a skipped layer, the cache
handed through untouched, and a hard error if a cache-writing pass is asked to skip -- but
as block wrappers rather than as a new `active_layers` argument threaded through
`LLaDAModel.forward`.

Why wrappers. Family B's model class is remote code that ships *inside the checkpoint*
(`models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/<rev>/modeling.py`). Editing it
means editing a revision-pinned file in the HF cache: invisible to the fork's git history,
clobbered by a re-download, and silently different if the revision moves. A wrapper touches
neither family's model file and gives one code path for both. Ported from this project's own
prototype at commit 0b6905c (`dllm_skip/`), with three changes:

  * `stub_kv` removed. Retracted by `08_audit_response.md` section 5: under DualCache a K/V
    stub written by a layer that never attends is never read, so the mode was bit-identical
    to `identity` at higher cost. Removing it also removes the prototype's most fragile code
    (its own RoPE/GQA reimplementation).
  * `reuse(m)` added -- the cached-residual-delta mode of `06` section 2.5 Stage 2.
  * arming is single-shot. The prototype left `controller.active` set until changed, so a
    forgotten update silently applied a stale skip set to the next pass. Here the controller
    counts wrapper invocations, ends the pass when it has seen every layer, and disarms:
    an un-armed pass runs at full depth. Forgetting fails safe and loudly-ish, not silently.

The invocation count also gives the cost accounting for free: every wrapper is called on
every pass, skipped or not, so `n_passes` is exact NFE and `layer_steps` is the executed
layer count that `04_efficiency.md` uses as the FLOP-side axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

IDENTITY, REUSE = "identity", "reuse"


class SkipError(RuntimeError):
    """Raised when a skip would corrupt the cache or read a delta that was never seeded."""


@dataclass
class SkipController:
    """Shared state consulted by every wrapped block. Armed for exactly one forward pass."""

    n_layers: int
    mode: str = IDENTITY
    _active: Optional[frozenset] = None       # None -> run every layer
    _seen: int = 0                            # wrapper invocations in the current pass
    deltas: Dict[int, torch.Tensor] = field(default_factory=dict)
    # accounting
    n_passes: int = 0
    layer_steps: int = 0
    n_skipped: int = 0
    history: List[Tuple[int, ...]] = field(default_factory=list)
    _pass_active: List[int] = field(default_factory=list)
    record_history: bool = True
    # Test affordance ONLY, for the positive control of `01` section 2 (Family B check 3),
    # which has to corrupt the cache on purpose to prove the other checks are sensitive.
    # Never set this in an experiment: `01` section 0 makes full-depth cache writes a rule.
    allow_cache_write_skip: bool = False

    # -- arming ---------------------------------------------------------------------------
    def arm(self, active: Optional[Iterable[int]], mode: Optional[str] = None) -> None:
        """Apply `active` to the NEXT forward pass only. None or all layers = full depth."""
        if self._seen:
            raise SkipError(f"arm() called mid-pass ({self._seen}/{self.n_layers} layers seen)")
        if mode is not None:
            if mode not in (IDENTITY, REUSE):
                raise SkipError(f"unknown mode {mode!r}")
            self.mode = mode
        self._active = None if active is None else frozenset(int(l) for l in active)

    def new_block(self) -> None:
        """Block boundary: cached deltas do not survive it (`01` section 2 check 4)."""
        self.deltas.clear()

    def reset_stats(self) -> None:
        self.n_passes = self.layer_steps = self.n_skipped = 0
        self.history.clear()

    # -- per-layer decision ---------------------------------------------------------------
    def _is_active(self, idx: int) -> bool:
        return self._active is None or idx in self._active

    def _tick(self, idx: int, executed: bool) -> None:
        if executed:
            self.layer_steps += 1
            if self.record_history:
                self._pass_active.append(idx)
        else:
            self.n_skipped += 1
        self._seen += 1
        if self._seen >= self.n_layers:           # pass complete
            self._seen = 0
            self.n_passes += 1
            if self.record_history:
                self.history.append(tuple(self._pass_active))
                self._pass_active = []
            self._active = None                   # single-shot: next pass is full depth


class _SkippableBase(nn.Module):
    def __init__(self, block: nn.Module, layer_idx: int, controller: SkipController):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.controller = controller

    def _skip_value(self, x: torch.Tensor) -> torch.Tensor:
        """The hidden state a skipped layer hands on."""
        c = self.controller
        if c.mode == IDENTITY:
            return x
        d = c.deltas.get(self.layer_idx)
        if d is None:
            raise SkipError(
                f"reuse mode skipped layer {self.layer_idx} with no seeded delta; the first "
                "refinement pass of each block must run at full depth (`01` section 1b)")
        return x + d

    def _store_delta(self, x_in: torch.Tensor, x_out: torch.Tensor) -> None:
        if self.controller.mode == REUSE:
            self.controller.deltas[self.layer_idx] = (x_out - x_in).detach()


class SkippableLLaDABlock(_SkippableBase):
    """Family A. `LLaDABlock.forward(x, attention_bias, layer_past, use_cache, replace_position)
    returns `(x, cache)`; the model loop asserts `cache is not None` when `use_cache`."""

    def forward(self, x, attention_bias=None, layer_past=None, use_cache=False,
                replace_position=None, **kw):
        c = self.controller
        if c._is_active(self.layer_idx):
            out = self.block(x, attention_bias=attention_bias, layer_past=layer_past,
                             use_cache=use_cache, replace_position=replace_position, **kw)
            self._store_delta(x, out[0])
            c._tick(self.layer_idx, True)
            return out
        if use_cache and layer_past is None and not c.allow_cache_write_skip:
            # A cache-writing pass: this layer has no K/V to hand back, and the loop would
            # trip `assert cache is not None`. By rule (`01` section 0) these run full depth.
            raise SkipError(
                f"layer {self.layer_idx} skipped on a cache-writing pass (layer_past is None); "
                "cache-writing passes run at full depth -- see `01` section 0")
        y = self._skip_value(x)
        c._tick(self.layer_idx, False)
        return (y, layer_past) if use_cache else (y, None)


class SkippableQwenLayer(_SkippableBase):
    """Family B. `Fast_dLLM_QwenDecoderLayer.forward(...)` returns a bare tensor, and takes
    `update_past_key_values` / `use_block_cache`, so a cache-writing pass is identified from
    the call itself rather than inferred."""

    def forward(self, hidden_states, *args, **kw):
        c = self.controller
        if c._is_active(self.layer_idx):
            out = self.block(hidden_states, *args, **kw)
            self._store_delta(hidden_states, out)
            c._tick(self.layer_idx, True)
            return out
        if (kw.get("update_past_key_values") or kw.get("use_block_cache")) \
                and not c.allow_cache_write_skip:
            # (i) prefill, (ii') in-block refresh and (iii) clean-block encode write the exact
            # cache; `01` section 1b keeps all three at full depth.
            raise SkipError(
                f"layer {self.layer_idx} skipped on a cache-writing pass "
                f"(update_past_key_values={kw.get('update_past_key_values')}, "
                f"use_block_cache={kw.get('use_block_cache')}) -- see `01` section 1b")
        y = self._skip_value(hidden_states)
        c._tick(self.layer_idx, False)
        return y


def _find_layers(model: nn.Module) -> Tuple[nn.ModuleList, type]:
    """Locate the block list and the wrapper class appropriate to it."""
    inner = getattr(model, "model", model)
    tr = getattr(inner, "transformer", None)
    if tr is not None and hasattr(tr, "blocks"):
        if getattr(inner.config, "block_group_size", 1) != 1:
            raise NotImplementedError("block_group_size != 1 is not supported")
        return tr.blocks, SkippableLLaDABlock
    if hasattr(inner, "layers"):
        return inner.layers, SkippableQwenLayer
    raise TypeError("could not find .transformer.blocks (Family A) or .layers (Family B)")


def install_skipping(model: nn.Module, controller: Optional[SkipController] = None,
                     mode: str = IDENTITY) -> SkipController:
    blocks, cls = _find_layers(model)
    n = len(blocks)
    controller = controller or SkipController(n_layers=n, mode=mode)
    if controller.n_layers != n:
        raise SkipError(f"controller says {controller.n_layers} layers, model has {n}")
    for i, blk in enumerate(blocks):
        blocks[i] = blk if isinstance(blk, _SkippableBase) else cls(blk, i, controller)
        blocks[i].controller = controller
    return controller


def uninstall_skipping(model: nn.Module) -> None:
    blocks, _ = _find_layers(model)
    for i, blk in enumerate(blocks):
        if isinstance(blk, _SkippableBase):
            blocks[i] = blk.block
