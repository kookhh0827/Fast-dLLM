"""Phase 3 amendment A25 (DLLM, results/phase3/PREREG.md): CUDA graphs for the refinement passes and the cache-writing
passes of Fast-dLLM v2 7B ("the graph stack"). The pass over the prompt stays eager.

Used only with `stage2_runner_v2.py --graphs`. Without that flag nothing here runs, and the eager path is that of
Block 1.

What stops a graph in the model code (revision 0661abf) and what this module does:
1. `DynamicCache` grows by `torch.cat` in each cache-writing pass, so the addresses of the cache change in each block.
   `StaticKV` keeps the keys and values of each layer in one buffer of the largest length. A cache-writing pass copies
   its 32 positions into the buffer, and the cache of each layer is a view of the first L positions.
   The values that the attention reads do not change: the refinement pass joins the view and the current block with
   `torch.cat` (a new contiguous tensor), and the sdpa function of transformers expands the keys and values of each
   KV head (`repeat_kv`, a copy) and makes them contiguous before the kernel.
2. The model makes the attention mask on the CPU in each pass (`eval_mask`) and copies it to the GPU. A graph cannot
   hold that copy. `eval_mask` keeps one copy on the GPU for each (rows, block size, cache length); the values are the
   same.
3. The shapes change with the cache length. One graph for each (pass type, cache length L, active layers). The runner
   captures all of them at the start of the cell, before the warm-up and the timed problems, for each L from the
   shortest prompt of the cell to the longest prompt plus the new tokens.
4. The Python code of the layer wrappers (dllm_skip.hook: the controller counts) and of the 4-bit kernels (the counts of
   each kernel path) does not run on a replay. After each replay, `forward` does the same counts.

A replay copies the 32 input tokens into the input of the graph, replays it, and returns the output tensor of the graph.
The sampler uses that output before the next pass (`torch.cat` and `argmax` make new tensors), so the next replay
cannot change a value that the sampler still reads.

Batch 1 only. No SoloQ (its patch is outside this check) and no block cache (`use_block_cache`, not in A7).
"""
import sys
import time
import types

import torch
from transformers.cache_utils import DynamicCache

BLOCK = 32


class StaticKV(DynamicCache):
    """A DynamicCache whose layers are views of fixed buffers (batch 1)."""

    def __init__(self, n_layers, n_kv_heads, head_dim, max_len, dtype, device):
        super().__init__()
        shape = (1, n_kv_heads, max_len, head_dim)
        self.kb = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.vb = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.max_len = max_len

    def reset(self):
        self.key_cache, self.value_cache = [], []
        self._seen_tokens = 0

    def set_length(self, n):
        """Each layer holds the first n positions (after a replay of a cache-writing pass)."""
        self.key_cache = [b[:, :, :n] for b in self.kb]
        self.value_cache = [b[:, :, :n] for b in self.vb]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        assert key_states.shape[0] == 1, "StaticKV: batch 1 only"
        n0 = self.key_cache[layer_idx].shape[-2] if len(self.key_cache) > layer_idx else 0
        n = n0 + key_states.shape[-2]
        if n > self.max_len:
            raise RuntimeError(f"StaticKV: {n} positions, the buffer holds {self.max_len}")
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        self.kb[layer_idx][:, :, n0:n].copy_(key_states)
        self.vb[layer_idx][:, :, n0:n].copy_(value_states)
        k, v = self.kb[layer_idx][:, :, :n], self.vb[layer_idx][:, :, :n]
        if len(self.key_cache) == layer_idx:
            self.key_cache.append(k)
            self.value_cache.append(v)
        else:
            self.key_cache[layer_idx], self.value_cache[layer_idx] = k, v
        return k, v


def _counters():
    """The dicts of the kernel-path counts (phase3_records.path_calls reads them)."""
    out = []
    gq = sys.modules.get("gptq_dequant")
    if gq is not None and hasattr(gq, "MarlinLinear"):
        out.append(gq.MarlinLinear.CALLS)
    pc = sys.modules.get("phase3_compress")
    if pc is not None:
        out.append(pc.KernelLinear.CALLS)
    return out


def _snap(ds):
    return [dict(d) for d in ds]


def _restore(ds, snap):
    for d, s in zip(ds, snap):
        d.clear()
        d.update(s)


class GraphStack:
    """Installs the graph stack on a model for one cell; `close()` takes it away."""

    def __init__(self, model, ctrl, sched, prompt_lens, max_new_tokens):
        from dllm_skip.hook import DepthB
        if "forward" in vars(model):
            raise SystemExit("--graphs: the model already has a forward of its own (a patch); not covered")
        self.model, self.ctrl = model, ctrl
        cfg = model.config
        dev = next(model.parameters()).device
        self.dev = dev
        lo = min(prompt_lens) // BLOCK * BLOCK
        hi = max(prompt_lens) // BLOCK * BLOCK
        if lo < BLOCK:
            raise SystemExit(f"--graphs: a prompt of {min(prompt_lens)} tokens (the graph stack needs 32 or more)")
        self.lengths = list(range(lo, hi + max_new_tokens + 1, BLOCK))
        self.kv = StaticKV(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads,
                           hi + max_new_tokens + 2 * BLOCK, torch.bfloat16, dev)
        # the masks: one GPU copy for each argument tuple (the values of the model's own function)
        inner = model.model
        self._eval_mask = inner.eval_mask
        cache = {}

        def eval_mask(seqlen, block_size, cache_seq_len):
            k = (int(seqlen), int(block_size), int(cache_seq_len))
            if k not in cache:
                cache[k] = self._eval_mask(*k).to(dev)
            return cache[k]
        inner.eval_mask = eval_mask
        self._orig = model.forward          # the bound method of the class (the instance has no forward of its own)
        self.ids = torch.zeros((1, BLOCK), dtype=torch.long, device=dev)
        self.graphs, self.outs, self.calls = {}, {}, {}
        self.replays = dict(refine=0, cache_write=0)
        self.eager = 0
        # the active layers of each pass type, as the sampler arms them (DepthB.arm; a static schedule)
        dep = DepthB(sched, ctrl, None, None, None)
        self.act = {}
        for kind, cw in (("refine", False), ("cache_write", True)):
            dep.arm(0.5, 0, 1, cw)
            self.act[kind] = ctrl._active
            ctrl.arm(None)
        self.capture()
        model.forward = self.forward

    def _run(self, kind, L):
        """One pass as the sampler calls it (the same arguments), on the cache of length L."""
        self.kv.set_length(L)
        self.ctrl.arm(self.act[kind])
        if kind == "refine":
            out = self._orig(input_ids=self.ids, use_cache=True, past_key_values=self.kv, update_past_key_values=False)
        else:
            out = self._orig(input_ids=self.ids, use_cache=True, past_key_values=self.kv, update_past_key_values=True,
                             block_size=BLOCK)
        return out.logits

    def capture(self):
        c = self.ctrl
        st = (c.n_passes, c.layer_steps, c.n_skipped, len(c.history))
        cnt = _counters()
        snap = _snap(cnt)
        torch.cuda.synchronize()
        self.mem0 = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
        t0 = time.perf_counter()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for L in self.lengths:
                for kind in ("refine", "cache_write"):
                    self._run(kind, L)                          # warm-up of this shape, eager
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        gib = lambda: (f"allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, reserved "     # noqa: E731
                       f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB")
        print(f"[graphs] {len(self.lengths)} lengths {self.lengths[0]} to {self.lengths[-1]}; after the warm-up: "
              f"{gib()}", flush=True)
        for n, L in enumerate(self.lengths):
            if n % 16 == 0:
                print(f"[graphs] capture at length {L}: {gib()}", flush=True)
            for kind in ("refine", "cache_write"):
                g = torch.cuda.CUDAGraph()
                before = _snap(cnt)
                with torch.cuda.graph(g):
                    self.outs[(kind, L)] = self._run(kind, L)
                # the kernel-path counts of one pass of this graph, added on each replay
                self.calls[(kind, L)] = [{p: d[p] - b.get(p, 0) for p in d} for d, b in zip(cnt, before)]
                self.graphs[(kind, L)] = g
        torch.cuda.synchronize()
        self.capture_s = time.perf_counter() - t0
        self.mem1 = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
        # the capture is not part of the cell: the counts go back to their values before it
        _restore(cnt, snap)
        c.n_passes, c.layer_steps, c.n_skipped = st[:3]
        del c.history[st[3]:]
        c._pass_active = []
        c.arm(None)
        self.kv.reset()

    def forward(self, input_ids=None, use_cache=None, past_key_values=None, update_past_key_values=False, **kw):
        if past_key_values is None:
            # the pass over the prompt (eager, on the static cache)
            self.eager += 1
            self.kv.reset()
            return self._orig(input_ids=input_ids, use_cache=use_cache, past_key_values=self.kv,
                              update_past_key_values=update_past_key_values, **kw)
        assert past_key_values is self.kv, "--graphs: a cache that is not the static cache"
        extra = set(kw) - ({"block_size"} if update_past_key_values else set())
        if extra or not use_cache or input_ids.shape != (1, BLOCK) or kw.get("block_size", BLOCK) != BLOCK:
            raise RuntimeError(f"--graphs: a pass that the graph stack does not cover ({sorted(kw)}, "
                               f"{tuple(input_ids.shape)})")
        kind = "cache_write" if update_past_key_values else "refine"
        L = self.kv.get_seq_length()
        if self.ctrl._active != self.act[kind]:
            raise RuntimeError(f"--graphs: a {kind} pass with other active layers than at capture")
        key = (kind, L)
        if key not in self.graphs:
            raise RuntimeError(f"--graphs: no graph for {key} (lengths {self.lengths[0]} to {self.lengths[-1]})")
        self.ids.copy_(input_ids)
        self.graphs[key].replay()
        self.replays[kind] += 1
        # the Python side of the pass: the controller counts and the kernel-path counts
        c = self.ctrl
        for i in range(c.n_layers):
            c._tick(i, c._is_active(i))
        for d, add in zip(_counters(), self.calls[key]):
            for p, v in add.items():
                d[p] = d.get(p, 0) + v
        if kind == "cache_write":
            self.kv.set_length(L + BLOCK)
        return types.SimpleNamespace(logits=self.outs[key], past_key_values=self.kv)

    def record(self):
        return dict(graphs=len(self.graphs), lengths=[self.lengths[0], self.lengths[-1]],
                    capture_s=self.capture_s, kv_buffer_len=self.kv.max_len,
                    mem_before=self.mem0, mem_after=self.mem1,
                    mem_graphs_reserved=self.mem1["reserved"] - self.mem0["reserved"],
                    replays=dict(self.replays), eager_passes=self.eager)

    def close(self):
        del self.model.forward
        self.model.model.eval_mask = self._eval_mask
        self.graphs.clear()
        self.outs.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
