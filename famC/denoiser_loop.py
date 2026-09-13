"""A line-for-line copy of `NemotronHTwoTowerForCausalLM._run_denoiser_step_diffusion` with two
optional attachments, installed on the model instance (the vendor file is not edited).

Why a copy: the vendor loop dispatches on `block.block_type` and calls the mixers through helper
functions, never `block(...)`, so a forward hook on a denoiser layer never fires. The copy exposes
the residual stream between layers:

  * `probe(layer_idx, block_type, hidden_in, hidden_out)` -- called after each layer (calibration);
  * `timer` -- a dict; when given, the layer loop is timed with CUDA synchronisation (Stage 0's f).

`null_check` runs the vendor method and the copy on the same inputs and requires bitwise-equal logits;
the step-3/4 driver refuses to run otherwise.
"""
import sys
import time
import types

import torch


def install(model, probe=None, timer=None):
    mod = sys.modules[type(model).__module__]
    _get_mod_params, _modulate = mod._get_mod_params, mod._modulate
    vendor = model._run_denoiser_step_diffusion

    def run(self, block_ids, cache_state, t=None, den_cache=None):
        ctx_len = cache_state["ctx_len"]                                      # noqa: F841 (as vendor)
        tower = self.denoiser_tower
        den_device = next(tower.parameters()).device
        den_input = block_ids.to(den_device)
        L = den_input.shape[1]                                                # noqa: F841 (as vendor)
        t_emb = None
        if t is not None:
            t_dev = t.to(device=den_device, dtype=self.dtype)
            t_repr = self.t_embedder(t_dev)
            t_emb = self.t_block(t_repr)
        if den_cache is None:
            den_cache = self._build_denoiser_cache_diffusion(cache_state, den_device)
        hidden = tower.embeddings(den_input)
        if timer is not None:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        for layer_idx, block in enumerate(tower.layers):
            residual = hidden
            if block.residual_in_fp32:
                residual = residual.to(torch.float32)
            m = None
            if t_emb is not None:
                m = _get_mod_params(t_emb, self.scale_shift_tables[layer_idx])
                shift, scale, gate = m
            if block.block_type in ("mamba", "attention"):
                h = hidden
                if m is not None:
                    h = _modulate(h, shift, scale)
                h = block.norm(h.to(dtype=block.norm.weight.dtype))
            else:
                h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
                if m is not None:
                    h = _modulate(h, shift, scale)
            if block.block_type == "mamba":
                d_conv = block.mixer.conv_kernel_size
                init_conv = den_cache.conv_states[layer_idx][..., -(d_conv - 1):]
                init_ssm = den_cache.ssm_states[layer_idx].contiguous()
                h = self._denoiser_block_mamba(block.mixer, h, init_conv, init_ssm)
            elif block.block_type == "attention":
                ctx_k = den_cache.key_cache[layer_idx]
                ctx_v = den_cache.value_cache[layer_idx]
                h = self._denoiser_block_attention(block.mixer, h, ctx_k, ctx_v)
            elif block.block_type in ["mlp", "moe"]:
                h = block.mixer(h)
            else:
                raise ValueError(f"Unknown block_type: {block.block_type}")
            if m is not None:
                h = gate.unsqueeze(1) * h
            new_hidden = residual + h
            if probe is not None:
                probe(layer_idx, block.block_type, hidden, new_hidden, den_input)
            hidden = new_hidden
        if timer is not None:
            torch.cuda.synchronize()
            timer.setdefault("layers_s", []).append(time.perf_counter() - t0)
        hidden = tower.norm_f(hidden)
        logits = self.lm_head(hidden.to(self.lm_head.weight.dtype)).float()
        return logits

    model._run_denoiser_step_diffusion = types.MethodType(run, model)
    return vendor


def null_check(model, vendor, block_ids, cache_state, t, den_cache):
    """Bitwise: the copy (with no attachments) against the vendor method."""
    with torch.no_grad():
        a = vendor(block_ids, cache_state, t=t, den_cache=den_cache)
        b = model._run_denoiser_step_diffusion(block_ids, cache_state, t=t, den_cache=den_cache)
    return bool(torch.equal(a, b)), float((a - b).abs().max())
