# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

from torch.cuda import nvtx

# --- depth-schedule plumbing (`01_experiment_plan.md` section 1) -------------------------
# Everything below is inert unless a schedule is passed: with `schedule=None` the three
# samplers take exactly the code path they took before, which is what keeps the Phase 0
# reproduction (results/phase0/baseline_*) valid.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                                   "..", "..")))
try:
    from dllm_skip.depth_schedule import StepState
except Exception:                                    # the fork can run without dllm_skip
    StepState = None


class GenStats(int):
    """NFE, plus the accounting `01` section 1 asks generate() to return.

    Subclasses int and carries NFE as its value, so every existing caller -- `out, nfe =
    generate(...)`, then `num_nfe += nfe` in eval_llada.py -- keeps working unchanged.
    """

    def __new__(cls, nfe, **kw):
        o = super().__new__(cls, int(nfe))
        o.nfe = int(nfe)
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def as_dict(self):
        return dict(nfe=self.nfe, layer_steps=self.layer_steps,
                    full_layer_steps=self.full_layer_steps, fallbacks=self.fallbacks,
                    depth_ratio=(self.layer_steps / self.full_layer_steps)
                    if self.full_layer_steps else 1.0)


class _Depth:
    """Arms the skip controller for one pass and records what it did.

    A no-op when no schedule is given, so the unscheduled path costs nothing. `layer_steps`
    is the executed (layer, pass) count and `full_layer_steps` what it would have been at
    full depth; their ratio is the depth ratio. Note this is NOT an iso-cost axis across
    cache modes (`08` section 2) -- wall clock is.
    """

    def __init__(self, schedule, controller, n_layers, log=None, sink=None,
                 protect_first_pass=False, skip_cache_writes=False):
        self.sched, self.ctrl, self.L, self.log = schedule, controller, n_layers, log
        self.sink = sink        # optional PassLog-like object: .add(rec, confidence, mask, committed)
        # PREREG section 7 diagnostics D1/D2. Neither is ever a gate input, and
        # `skip_cache_writes` in particular BREAKS the full-depth cache-write rule of
        # `01` section 0 on purpose -- it is the {approximated} arm of E-1. Both default off.
        self.protect_first_pass = bool(protect_first_pass)
        self.skip_cache_writes = bool(skip_cache_writes)
        if self.skip_cache_writes:
            # THREE independent guards enforce the full-depth cache-write rule, and D2 has to
            # lift all three or it silently measures the ordinary cell. Clearing only this
            # class's `force_full` is what the first D2 run did, and it reported layer_steps
            # identical to the unmodified prefix k=6 cell -- the schedule had already returned
            # every layer before the controller was ever armed.
            if controller is not None:
                controller.allow_cache_write_skip = True          # 1. the hook's guard
            if schedule is not None:
                schedule.full_depth_on_cache_write = False        # 2. the schedule's guard
            # 3. `force_full` from the caller, cleared per pass in `arm`
        self.layer_steps = self.full_layer_steps = self.fallbacks = 0
        self.last = None        # the record just appended, so the sampler can attach to it

    @property
    def on(self):
        return self.sched is not None and self.ctrl is not None

    def arm(self, state, force_full=False):
        self.full_layer_steps += self.L
        if self.ctrl is not None:
            # Every pass may seed. A block-start pass covers the whole canvas, so the hook
            # slices its delta to the block's rows (`01` section 1b, 2026-09-09) -- that is the
            # previous pass's delta for those positions, staleness 1, ordinary reuse. Family A
            # therefore adds NO extra full-depth pass and reads `reuse` against the same
            # L_eq_req as every other mode.
            self.ctrl.store_deltas = True
            self.ctrl.block_slice = state.block_slice
        # D1: hold the block's first refinement pass at full depth (no cache in this arm, so
        # this is the ONLY full-depth pass a block gets -- the thing the cell isolates).
        if self.on and self.protect_first_pass and not state.is_cache_write \
                and state.step_in_block <= 0:
            force_full = True
        # D2: let the cache-writing passes run shallow, against the rule.
        if self.on and self.skip_cache_writes and state.is_cache_write:
            force_full = False
        # reuse(m): every m-th refinement pass recomputes the deltas at full depth
        m = getattr(self.ctrl, "reuse_m", None)
        if (self.on and getattr(self.ctrl, "mode", None) == "reuse" and m
                and not state.is_cache_write and state.step_in_block % m == 0):
            force_full = True
        act = None
        if self.on and not force_full:
            act = tuple(self.sched.active_layers(state))
            self.ctrl.arm(act)
        elif self.on:
            self.ctrl.arm(None)
        self.layer_steps += self.L if act is None else len(act)
        rec = dict(block=state.block_idx, step=state.step_in_block,
                   mask_ratio=round(float(state.mask_ratio), 4),
                   cache_write=bool(state.is_cache_write),
                   depth=self.L if act is None else len(act))
        if self.log is not None:
            self.log.append(rec)
        self.last = rec
        return act


def _state(mask_ratio, block_idx, num_blocks, step_in_block, is_cache_write,
           block_slice=None):
    if StepState is None:
        raise RuntimeError("dllm_skip.depth_schedule is not importable; cannot use a schedule")
    return StepState(mask_ratio=float(mask_ratio), block_idx=int(block_idx),
                     num_blocks=int(num_blocks), step_in_block=int(step_in_block),
                     is_cache_write=bool(is_cache_write), block_slice=block_slice)


def _n_layers(model):
    inner = getattr(model, "model", model)
    tr = getattr(inner, "transformer", None)
    return len(tr.blocks) if tr is not None and hasattr(tr, "blocks") else len(inner.layers)


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# def get_num_transfer_tokens(mask_index, steps):
#     '''
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.

#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     '''
#     mask_num = mask_index.sum(dim=1, keepdim=True)

#     base = mask_num // steps
#     remainder = mask_num % steps

#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1

#     return num_transfer_tokens

def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode='floor')  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens



@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None,
             schedule=None, controller=None, fallback_conf=None, log=None,
             protect_first_pass=False, skip_cache_writes=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    dep = _Depth(schedule, controller, _n_layers(model), log,
                 protect_first_pass=protect_first_pass,
                 skip_cache_writes=skip_cache_writes)
    if fallback_conf is not None:
        raise NotImplementedError(
            "the confidence fallback is implemented for generate_with_dual_cache only")
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            if dep.on:      # no cache is ever written here, so every pass may be shallow
                bs_, be_ = prompt.shape[1] + num_block * block_length, prompt.shape[1] + (num_block + 1) * block_length
                dep.arm(_state((x[:, bs_:be_] == mask_id).float().mean().item(),
                               num_block, num_blocks, i, False))
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, GenStats(nfe, layer_steps=dep.layer_steps,
                       full_layer_steps=dep.full_layer_steps,
                       fallbacks=dep.fallbacks, steps_log=log)



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None,
             schedule=None, controller=None, fallback_conf=None, log=None,
             protect_first_pass=False, skip_cache_writes=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    dep = _Depth(schedule, controller, _n_layers(model), log,
                 protect_first_pass=protect_first_pass,
                 skip_cache_writes=skip_cache_writes)
    if fallback_conf is not None:
        raise NotImplementedError(
            "the confidence fallback is implemented for generate_with_dual_cache only -- "
            "the deployed regime `06` section 2.5 states the bar for")

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        if dep.on:                                  # cache-writing pass: full depth by rule
            dep.arm(_state(block_mask_index.float().mean().item(), num_block, num_blocks, 0,
                           True), force_full=True)
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            if dep.on:
                dep.arm(_state((x[:, current_block_start:current_block_end] == mask_id)
                               .float().mean().item(), num_block, num_blocks, i, False))
            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1

    return x, GenStats(nfe, layer_steps=dep.layer_steps,
                       full_layer_steps=dep.full_layer_steps,
                       fallbacks=dep.fallbacks, steps_log=log)

@torch.no_grad()
def generate_with_dual_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None,
    schedule=None, controller=None, fallback_conf=None, log=None, sink=None,
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0
    dep = _Depth(schedule, controller, _n_layers(model), log, sink)

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block.
        #    Cache-writing pass: full depth by rule (`01` section 0), enforced here as well
        #    as by the hook's own guard.
        if dep.on:
            dep.arm(_state(block_mask_index.float().mean().item(), nb, num_blocks, 0, True,
                           block_slice=(s, e)), force_full=True)
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        global_mask_index = (x == mask_id)
        # Do not touch beyond current block in this phase
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            if dep.on:
                r_i = (x[:, s:e] == mask_id).float().mean().item()
                dep.arm(_state(r_i, nb, num_blocks, i, False, block_slice=None))
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits  # shape expected by get_transfer_index*

            # Confidence fallback (`01` section 1). Opt-in: with fallback_conf=None nothing
            # here runs, which matters because a per-pass softmax over the 126k vocabulary is
            # exactly the control overhead `04` section 1 says must clear phi > ov/f. Stage 2
            # runs without it on purpose (`06` section 2.5).
            if dep.on and fallback_conf is not None:
                m_blk = (x[:, s:e] == mask_id)
                if m_blk.any():
                    conf = logits_blk.float().softmax(-1).max(-1).values
                    if conf[m_blk].max().item() < fallback_conf:
                        dep.fallbacks += 1
                        dep.arm(_state(r_i, nb, num_blocks, i, False), force_full=True)
                        logits_blk = model(
                            x[:, s:e], past_key_values=past_key_values, use_cache=True,
                            replace_position=replace_position
                        ).logits

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, s:e] == mask_id)  # (B, block_length)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                # `return_confidence` costs nothing: the softmax already ran inside the call
                # (`01` section 1, 2026-09-09). Off by default, so every other caller is
                # byte-identical.
                want_conf = dep.sink is not None
                res = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold,
                    return_confidence=want_conf
                )
                if want_conf:
                    x0_blk, transfer_idx_blk, conf_blk = res
                    dep.sink.add(dep.last, conf_blk, mask_blk, transfer_idx_blk)
                else:
                    x0_blk, transfer_idx_blk = res
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            # Merge back into x[:, s:e] using torch.where (no masked slice assignment)
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)  # static concatenation

            nfe += 1

    return x, GenStats(nfe, layer_steps=dep.layer_steps,
                       full_layer_steps=dep.full_layer_steps,
                       fallbacks=dep.fallbacks, steps_log=log)



def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
    return_confidence: bool = False,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
        confidence: (B, L) float64, only when return_confidence — the tensor the threshold rule
            compared. Masked positions hold their max-probability; every other position holds
            `torch.finfo(float64).min`, the dtype's most negative FINITE value, not -inf.

    `return_confidence` (2026-09-09, `01` §1) exists so PREREG §5's per-pass record uses the
    sampler's own value instead of a second softmax. The softmax is already inside the timed
    baseline (0.26 ms of a 21.01 ms refinement pass, the `transfer` column of
    results/phase0/profile.md), so the record costs nothing. The commit rule is untouched, and
    with the flag off every call site is byte-identical.
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked (consider fully unmasked rows)
        transfer_index = transfer_index & mask_index

        return (x0, transfer_index, confidence) if return_confidence else (x0, transfer_index)

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return (x0, transfer_index, confidence) if return_confidence else (x0, transfer_index)

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue
        
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    # model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    
        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
