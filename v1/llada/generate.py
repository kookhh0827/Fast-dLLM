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

import math
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
        self.skipped_last = False

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
        # Phase 0.3 needs this: tau_r applies only where the depth was actually cut, so a
        # regime's protected passes keep tau_w. Two different things make a pass run the full
        # stack and they must not be conflated: `act is None` is a forced-full pass (a cache
        # write, or a reuse refresh), while a RegimeSchedule returns every layer on a pass its
        # regime protects. The k = 0 baseline row is neither -- it "skips" an empty set on every
        # refinement pass and must sweep tau_r like the depth rows, because PREREG 0.3 section 2
        # requires the baseline and the depth rows to move the same parameter.
        intends_skip = True
        if self.on:
            f = getattr(self.sched, "skips_this_pass", None)
            if f is not None:
                intends_skip = bool(f(state))
        self.skipped_last = (act is not None) and intends_skip
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
             protect_first_pass=False, skip_cache_writes=False,
             sink=None, timer=None, block_softmax=False):
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

    Phase 2 P7b (front N32, no cache): `sink` records each pass like the DualCache sampler (a PassLog; it needs a
    schedule, as `--cells full` gives), `timer` records a CUDA event at the start of each pass (kind "refine"), and
    `block_softmax` computes the commit on the block slice only. Only the block can commit, so the commits are the
    same; the float64 softmax over the whole canvas is not paid. With the three left at their defaults, the path is
    the old one.
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
            _mark(timer, "refine")
            nfe += 1
            mask_index = (x == mask_id)
            if dep.on:      # no cache is ever written here, so every pass may be shallow
                bs_, be_ = prompt.shape[1] + num_block * block_length, prompt.shape[1] + (num_block + 1) * block_length
                dep.arm(_state((x[:, bs_:be_] == mask_id).float().mean().item(),
                               num_block, num_blocks, i, False))
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            want_conf = sink is not None and dep.on
            if block_softmax or want_conf:
                assert factor is None, "the P7b options need the threshold or quota rule"
                bs_, be_ = prompt.shape[1] + num_block * block_length, prompt.shape[1] + (num_block + 1) * block_length
                sl = slice(bs_, be_) if block_softmax else slice(None)
                res = get_transfer_index(logits[:, sl], temperature, remasking, mask_index[:, sl], x[:, sl],
                                         num_transfer_tokens[:, i] if threshold is None else None, threshold,
                                         return_confidence=want_conf)
                if want_conf:
                    sink.add(dep.last, res[2], mask_index[:, sl], res[1], threshold)
                x[:, sl] = torch.where(res[1], res[0], x[:, sl])
            elif factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[transfer_index] = x0[transfer_index]
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
                x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    _mark(timer, "end")
    return x, GenStats(nfe, layer_steps=dep.layer_steps,
                       full_layer_steps=dep.full_layer_steps,
                       fallbacks=dep.fallbacks, steps_log=log)


# ---------------------------------------------------------------------------------------------------------------
# DC-Leap: ported for Phase 2 P7b (results/phase2/PREREG_week2.md, amendment A10) from llada/generate.py of
# https://github.com/ffh-wyls/DC-Leap, commit e105d45: get_top1_info, verify_and_commit, generate_with_dc_leap.
# The notice of the authors:
#
# MIT License
#
# Copyright (c) 2026 ffh-wyls
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ---------------------------------------------------------------------------------------------------------------

def get_top1_info(logits: torch.Tensor):
    probs = F.softmax(logits, dim=-1)
    top1_probs, top1_indices = torch.max(probs, dim=-1)
    return top1_indices, top1_probs


def verify_and_commit(
    region_logits: torch.Tensor,
    region_x: torch.Tensor,
    mask_id: int,
    commit_thres: float,
    left_boundary_known: bool
):
    top1_indices, top1_probs = get_top1_info(region_logits)

    is_confident = (top1_probs > commit_thres)

    if left_boundary_known:
        contiguity_mask = torch.cumprod(is_confident.int(), dim=0).bool()
    else:
        contiguity_mask = torch.zeros_like(is_confident, dtype=torch.bool)

    is_mask = (region_x == mask_id)
    final_commit_mask = contiguity_mask & is_mask

    return top1_indices, final_commit_mask


@torch.no_grad()
def generate_with_dc_leap(
    model,
    prompt: torch.Tensor,
    steps: int,
    commit_thres: float,
    draft_thres: float,
    gen_length: int,
    block_length: int,
    max_window_size: int,
    cfg_scale: float,
    temperature: float,
    remasking='low_confidence',
    mask_id: int = 126336,
    schedule=None, controller=None, log=None, timer=None,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        commit_thres: Confidence threshold for Dynamic Contiguous Verification (DCV). Only the longest contiguous prefix within the decoding window where tokens exceed this threshold will be committed.
        draft_thres: Confidence threshold  for the Draft Mechanism. High-confidence tokens predicted outside the decoding window are cached to provide look-ahead context for bidirectional attention.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        max_window_size: Maximum size (L) of the dynamic decoding window.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.

    Changes for our harness (the interface only; the decoding is the authors'): the function returns (x, GenStats)
    with the NFE, as the other samplers of this file; `schedule` and `controller` arm the skip controller for each
    pass (the k = 0 schedule of `--cells full`, so every cell pays the same instrumentation); `log` gets one record
    for each pass (pass index, verified end before the pass, commits, and whether the pass used the floor commit);
    `timer` gets a CUDA event at the start of each pass. These records use only values that the loop already moves
    to the host, so they add no sync. `steps` and `block_length` are not read, as in the authors' function.
    '''
    device = model.device
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt
    prompt_len = prompt.shape[1]
    draft_bank = torch.full((gen_length,), mask_id, dtype=torch.long, device=device)
    nfe = 0
    dep = _Depth(schedule, controller, _n_layers(model), None)

    verified_end = 0
    while verified_end < gen_length:

        l2r_len = max_window_size
        future_len = max_window_size
        win_s = verified_end
        win_e = min(verified_end + l2r_len + future_len, gen_length)

        abs_win_s = prompt_len + win_s
        abs_win_e = prompt_len + win_e

        if abs_win_s >= abs_win_e: break

        _mark(timer, "refine")
        if dep.on:
            dep.arm(_state((gen_length - verified_end) / gen_length, 0, 1, nfe, False))
        rec = dict(step=nfe, verified_end=verified_end, committed=0, floor=False) if log is not None else None
        nfe += 1

        x_for_prediction = x.clone()

        drafts = draft_bank[win_s:win_e]
        target_slice = x_for_prediction[0, abs_win_s:abs_win_e]
        mask_locs = (target_slice == mask_id)
        valid_drafts = (drafts != mask_id)
        fill_locs = mask_locs & valid_drafts
        if fill_locs.any():
            x_for_prediction[0, abs_win_s:abs_win_e][fill_locs] = drafts[fill_locs]

        leader_end = min(abs_win_s + max_window_size, abs_win_e)
        x_for_prediction[0, abs_win_s:leader_end] = mask_id

        if cfg_scale > 0.:
            pred_model_out = model(x_for_prediction, output_hidden_states=False)
            prediction_logits = pred_model_out.logits
            pred_conditional, pred_unconditional = prediction_logits.chunk(2, dim=0)
            prediction_logits = pred_unconditional + (cfg_scale + 1) * (pred_conditional - pred_unconditional)
        else:
            pred_model_out = model(x_for_prediction, output_hidden_states=False)
            prediction_logits = pred_model_out.logits

        l2r_abs_end = min(prompt_len + verified_end + l2r_len, abs_win_e)

        if l2r_abs_end > abs_win_s:
            region_logits = prediction_logits[0, abs_win_s:l2r_abs_end]
            region_x = x[0, abs_win_s:l2r_abs_end]

            left_boundary_known = True
            if verified_end > 0:
                left_boundary_known = (x[0, abs_win_s - 1].item() != mask_id)

            top1_tokens, commit_mask = verify_and_commit(
                region_logits, region_x, mask_id,
                commit_thres, left_boundary_known
            )

            if not commit_mask.any():
                is_mask = (region_x == mask_id)
                if is_mask.any():
                    first_idx = is_mask.nonzero(as_tuple=True)[0][0]
                    commit_mask[first_idx] = True
                    if rec is not None:
                        rec["floor"] = True

            if commit_mask.any():
                update_idx = commit_mask.nonzero(as_tuple=True)[0]
                x[0, abs_win_s + update_idx] = top1_tokens[update_idx]
                if rec is not None:
                    rec["committed"] = int(update_idx.numel())

                next_mask = (x[0, abs_win_s:l2r_abs_end] == mask_id).nonzero(as_tuple=True)
                if next_mask[0].numel() > 0:
                    verified_end += next_mask[0][0].item()
                else:
                    verified_end += (l2r_abs_end - abs_win_s)

        draft_logits = prediction_logits[0, abs_win_s:abs_win_e]
        r_idx, r_probs = get_top1_info(draft_logits)

        draft_candidates_mask = r_probs > draft_thres
        if draft_candidates_mask.any():
            indices = draft_candidates_mask.nonzero(as_tuple=True)[0]
            bank_indices = win_s + indices

            valid = bank_indices < gen_length
            if valid.all():
                draft_bank[bank_indices] = r_idx[indices]
            elif valid.any():
                draft_bank[bank_indices[valid]] = r_idx[indices][valid]
        if rec is not None:
            log.append(rec)
    x = x[:, :prompt.shape[1] + gen_length]
    _mark(timer, "end")
    return x, GenStats(nfe, layer_steps=dep.layer_steps, full_layer_steps=dep.full_layer_steps,
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
def dus_levels(block_length, base, skip_exp=1):
    """DUS's dilated unmasking schedule for one block, as within-block offsets.

    Ported verbatim from github.com/omerlux/DUS (MIT) `generate.py`: `dilated_unmask_levels`
    followed by `merge_last_level`. Positions are a function of (block_length, base, skip_exp) only;
    nothing is read from the model. `skip_exp` is DUS's `base_skip`, the exponent of the first
    stride -- a positional parameter; the confidence-reading part of DUS is the separate
    `confidence_threshold` remasking, which this port does not have.
    """
    if base < 1 or skip_exp < 1:
        raise ValueError("base and skip_exp must be >= 1")
    if base == 1:
        return [list(range(block_length))]
    stride = block_length // (base ** skip_exp)
    levels, revealed = [], set()
    while stride >= 1:
        this_round = [i for i in range(block_length) if i % stride == 0 and i not in revealed]
        if this_round:
            levels.append(this_round)
            revealed.update(this_round)
        stride //= base
    remainder = [i for i in range(block_length) if i not in revealed]
    if remainder:
        levels.append(remainder)
    if len(levels) >= 2 and len(levels[-1]) < len(levels[-2]):
        levels[-2].extend(levels[-1])
        levels.pop()
    return levels


class CaiRule:
    """The commit rule of CAI-DLLM (arXiv 2608.22646) for Phase 2 P7b (results/phase2/PREREG_week2.md, amendment
    A10). Our own code, one instance for each block, batch 1. It is called once for each pass of the block: step 0 is
    the cache-writing pass, steps 1 to T - 1 the refinement passes (T = the block length, 64).

    The paper: the threshold schedule (Box 3), the end threshold of each block (Box 4), the position factors (Eq. 6,
    Box 6), the step budgets (Eq. 5), the forced commit (Eq. 7) and the grind phase (Box 7). Where the paper and the
    authors' code differ, this follows the code (github.com/flarebrienne/CAI-DLLM, commit 6246dad, the LLaDA GSM8K
    setting `--use_cai --cai_mode apd_per_block` with gating on), as A10 asks. The differences are in
    docs/papers/cai_dllm.md, section 2.1:
      1. the grind threshold is a count of 1.5 commits per pass, not a share of the block;
      2. the passes before step 15 do not count toward the grind phase;
      3. at the grind trigger, the masked tokens above 0.40 commit once, and the block continues;
      4. the threshold and the grind commit use ">", not ">=";
      5. the cache is written only at the block start (our DualCache).
    Also from the code (the paper does not say): the confidence is a float32 softmax at the argmax; the floor commits
    the most confident masked token (topk); a block has at most T passes; blocks after the fourth reuse theta_e 0.40.
    Left out (A10): the end-token delay, because the authors' baselines use it too.
    """
    THETA_S, WARMUP, THETA_E = 0.90, 0.15, (0.70, 0.60, 0.50, 0.40)
    CLIP = (0.20, 0.98)
    EASY, HARD, BUDGET = 0.50, 0.25, (8, 32, 64)
    GRIND_N, GRIND_K, GRIND_MIN_STEP, GRIND_CONF = 1.5, 4, 15, 0.40

    def __init__(self, block_idx, block_length, device):
        self.T = block_length
        self.theta_e = self.THETA_E[min(block_idx, len(self.THETA_E) - 1)]
        self.rel = torch.arange(block_length, device=device).unsqueeze(0)
        scale = torch.full((1, block_length), 1.10, dtype=torch.float32, device=device)
        scale[self.rel < 48] = 1.05
        scale[self.rel < 32] = 0.90
        scale[self.rel < 16] = 0.80
        self.scale = scale
        self.budget = None
        self.low, self.fired = 0, False

    def theta(self, t):
        """Box 3: theta_s during the warm-up, then a cosine from theta_s to theta_e."""
        w = self.WARMUP * self.T
        if t < w:
            return self.THETA_S
        progress = (t - w) / (self.T - w)
        return self.theta_e + (self.THETA_S - self.theta_e) * (0.5 * (1.0 + math.cos(math.pi * progress)))

    def _init_budget(self, conf0, masked):
        """Eq. 5 at step 0: a tier from the confidence, then the position moves a token to a later tier."""
        tier = torch.ones_like(conf0, dtype=torch.long)
        tier[conf0 > self.EASY] = 0
        tier[conf0 < self.HARD] = 2
        tier[(self.rel >= 32) & (self.rel < 48) & (tier == 0)] = 1
        tier[(self.rel >= 48) & (tier <= 1)] = 2
        b = torch.tensor(self.BUDGET, device=conf0.device)[tier]
        self.budget = torch.where(masked, b, torch.zeros_like(b))

    def commit(self, logits, masked, t):
        """One pass. logits (1, T, V) of the block, masked (1, T). Returns the argmax tokens, the commits, the
        confidence (0 where not masked), the counts [above the threshold, forced by the budget only, forced by the
        grind phase only, floor only] as a tensor, and the number of commits."""
        x0 = torch.argmax(logits, dim=-1)
        p = F.softmax(logits.float(), dim=-1)
        conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1).masked_fill(~masked, 0.0)
        if t == 0:
            self._init_budget(conf, masked)
        above = conf > (self.theta(t) * self.scale).clamp(*self.CLIP)
        by_budget = (t >= self.budget) & masked & (self.budget > 0)
        by_grind = torch.zeros_like(masked)
        if self.low >= self.GRIND_K and not self.fired:
            by_grind = masked & (conf > self.GRIND_CONF)
            self.fired = True
        com = above | by_budget | by_grind
        floor = torch.zeros_like(com).scatter_(-1, torch.topk(conf, k=1, dim=-1).indices, True)
        com = (com | floor) & (conf > 0.0)
        n = int(com.sum())
        if t < self.GRIND_MIN_STEP:
            self.low = 0
        elif n < self.GRIND_N:
            self.low += 1
        else:
            self.low = 0
        counts = torch.stack([above.sum(), (by_budget & ~above).sum(), (by_grind & ~above & ~by_budget).sum(),
                              (com & ~(above | by_budget | by_grind)).sum()])
        return x0, com, conf, counts, n


def generate_with_dual_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, tau_r=None, factor=None,
    schedule=None, controller=None, fallback_conf=None, log=None, sink=None, dus_base=None, timer=None,
    cai=False, cai_log=None, cai_probe=None,
):
    """`cai` (Phase 2 P7b): commit by `CaiRule` (CAI-DLLM) on every pass instead of the threshold rule; batch 1.
    `cai_log` gets (pass record, theta of the step, counts tensor) for each pass; the caller moves the counts to the
    host after the sampler returns. `cai_probe(block, step, block_start, logits, masked, commits, tokens)` is called
    after each commit (check I2 of A10).

    `timer` (Phase 2 P5): a list. If given, a CUDA event is recorded at the start of each pass, as
    (kind, event) with kind "cache_write" or "refine", and one ("end", event) after the last pass. The time of
    a pass is the interval to the next start, so the two kinds sum to the decoding time. No sync is added.

    `dus_base` (Phase 1.5): commit by DUS's planned schedule instead of the threshold rule. The
    block-start pass (cache-writing, full depth) commits level 0 and each refinement pass the next
    level, argmax at those positions; the number of passes per block is the number of levels, fixed
    before decoding. Confidence is still computed when a sink records passes, and never decides a
    commit."""
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
        if dus_base is not None:
            dus = dus_levels(block_length, dus_base)
            dus_mask = []
            for lv in dus:
                mk = torch.zeros((B, block_length), dtype=torch.bool, device=x.device)
                mk[:, lv] = True
                dus_mask.append(mk)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block.
        #    Cache-writing pass: full depth by rule (`01` section 0), enforced here as well
        #    as by the hook's own guard.
        _mark(timer, "cache_write")
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

        if cai:
            assert B == 1 and dus_base is None and factor is None and fallback_conf is None, "CAI-DLLM: batch 1 only"
            rule = CaiRule(nb, block_length, x.device)
            mask_blk0 = (x[:, s:e] == mask_id)
            x0_blk, tr_blk, conf_blk, cnt, _ = rule.commit(out_full.logits[:, s:e], mask_blk0, 0)
            if cai_probe is not None:
                cai_probe(nb, 0, s, out_full.logits[:, s:e], mask_blk0, tr_blk, x0_blk)
            if dep.sink is not None and dep.on:
                dep.sink.add(dep.last, conf_blk, mask_blk0, tr_blk, None)
            if cai_log is not None:
                cai_log.append((dep.last, rule.theta(0), cnt))
            x0 = x.clone()
            x0[:, s:e] = x0_blk
            transfer_index = torch.zeros_like(global_mask_index)
            transfer_index[:, s:e] = tr_blk
        elif dus_base is not None:
            x0, _ = get_transfer_index(out_full.logits, temperature, remasking, global_mask_index, x,
                                       None, 2.0)          # x0 only; nothing clears a 2.0 threshold
            level0 = torch.zeros_like(global_mask_index)
            level0[:, s:e] = dus_mask[0]
            transfer_index = level0 & global_mask_index
        elif factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            # Phase 2 amendment A3: the cache-writing pass is recorded like a refinement pass
            # (committed, positions >= tau, mean confidence). The commit rule is untouched.
            want_conf = dep.sink is not None and dep.on
            if B == 1:
                res = get_transfer_index(
                    out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold,
                    return_confidence=want_conf
                )
                x0, transfer_index = res[0], res[1]
                if want_conf:
                    dep.sink.add(dep.last, res[2], global_mask_index, transfer_index, threshold)
            else:
                # Phase 2 P5 (batch > 1): only the block [s, e) can commit here, because global_mask_index is False
                # elsewhere. So the float64 softmax runs on the block slice; over the whole canvas it is
                # B x length x vocabulary in float64 (about 33 GB at batch 32). The commits are the same.
                res = get_transfer_index(
                    out_full.logits[:, s:e], temperature, remasking, global_mask_index[:, s:e], x[:, s:e], quota0,
                    threshold, return_confidence=want_conf
                )
                x0 = x.clone()
                x0[:, s:e] = res[0]
                transfer_index = torch.zeros_like(global_mask_index)
                transfer_index[:, s:e] = res[1]
                if want_conf:
                    dep.sink.add(dep.last, res[2], global_mask_index[:, s:e], res[1], threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block if dus_base is None else len(dus)):
            # Evaluate logits only for current block with cache
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            _mark(timer, "refine")
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

            if cai:
                x0_blk, transfer_idx_blk, conf_blk, cnt, _ = rule.commit(logits_blk, mask_blk, i)
                if cai_probe is not None:
                    cai_probe(nb, i, s, logits_blk, mask_blk, transfer_idx_blk, x0_blk)
                if dep.sink is not None:
                    dep.sink.add(dep.last, conf_blk, mask_blk, transfer_idx_blk, None)
                if cai_log is not None:
                    cai_log.append((dep.last, rule.theta(i), cnt))
            elif dus_base is not None:
                want_conf = dep.sink is not None
                res = get_transfer_index(logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, 2.0,
                                         return_confidence=want_conf)
                x0_blk = res[0]
                transfer_idx_blk = dus_mask[i] & mask_blk               # the plan, not the logits
                if want_conf:
                    dep.sink.add(dep.last, res[2], mask_blk, transfer_idx_blk)
            elif factor is None:
                # Phase 0.3's knob. tau_w -- the threshold on the block-start cache-writing pass
                # above -- stays `threshold` in every cell: those passes are full depth
                # everywhere, their confidence is not deflated, and the baseline row and the
                # depth rows must move the same parameter (PREREG 0.3 section 2). A regime cell
                # applies tau_r only where it actually cut the depth.
                thr_i = threshold
                if tau_r is not None and (not dep.on or dep.skipped_last):
                    thr_i = tau_r
                quota_i = None if thr_i is not None else num_transfer_tokens[:, i]  # (B,)
                # `return_confidence` costs nothing: the softmax already ran inside the call
                # (`01` section 1, 2026-09-09). Off by default, so every other caller is
                # byte-identical.
                want_conf = dep.sink is not None
                res = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, thr_i,
                    return_confidence=want_conf
                )
                if want_conf:
                    x0_blk, transfer_idx_blk, conf_blk = res
                    dep.sink.add(dep.last, conf_blk, mask_blk, transfer_idx_blk, thr_i)
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

    _mark(timer, "end")
    return x, GenStats(nfe, layer_steps=dep.layer_steps,
                       full_layer_steps=dep.full_layer_steps,
                       fallbacks=dep.fallbacks, steps_log=log)


def _mark(timer, kind):
    """Phase 2 P5: record a CUDA event for the start of a pass (no sync)."""
    if timer is not None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        timer.append((kind, ev))


def pass_times(timer):
    """Seconds and counts of the cache-writing and refinement passes of one `timer` list. Call after a sync."""
    t, n = dict(cache_write=0.0, refine=0.0), dict(cache_write=0, refine=0)
    for (k, a), (_, b) in zip(timer, timer[1:]):
        t[k] += a.elapsed_time(b) / 1000.0
        n[k] += 1
    return t, n


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
