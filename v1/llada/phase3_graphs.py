"""Phase 3 amendment A25 (DLLM, results/phase3/PREREG.md): CUDA graphs for the DualCache passes of LLaDA-8B-Instruct
("the graph stack"). The cache-writing passes and the refinement passes replay graphs.

Used only with `stage2_runner.py --graphs`. Without that flag nothing here runs, and the eager path is that of
Block 1. The model code does not change: each graph runs the same code as the eager pass.

What stops one graph for all passes in the model code and what this module does:
1. The cache-writing pass of a block runs over the whole sequence (prompt and generation: L positions) with no cache,
   and returns the keys and values of each layer. The refinement pass of the block that starts at position s writes
   its 32 keys and values into that cache at [s, s + 32) and applies RoPE up to s + 32. The model takes s from the
   memoised span of `replace_position` (Python integers). So the shapes depend on L, and the code depends on s.
   One graph for the cache-writing pass of each L, and one graph for the refinement pass of each (L, s).
2. The refinement graphs of L take the keys and values that the cache-writing graph of L returns as their cache. A
   replay writes into them in place, as the eager pass does. The next cache-writing replay writes all of them again.
3. L is the length of one problem (the prompt plus the new tokens). The runner calls `prepare(Lp)` before each
   problem, outside its timer. It captures the graphs of that L, unless they exist; it keeps the graphs of the last
   `keep` lengths. All graphs of one L share one memory pool: they run one after the other, never at the same time.
   The time of the captures is recorded (`record()`), and it is not in the time of a problem.
4. The Python code of the layer wrappers (dllm_skip.hook: the controller counts) and of the 4-bit kernels (the counts
   of each kernel path) does not run on a replay. After each replay, `forward` does the same counts.

The capture and the replays run under torch.no_grad(), as the sampler does (stage2_runner.decode_batch).

A replay returns the output tensors of the graph. The sampler uses them before the next replay of the same graph:
the logits of a cache-writing pass before the refinement passes of its block, and the logits of a refinement pass
before the next pass (`torch.where`, `torch.cat` and the commit rule make new tensors).

Batch 1 only, the DualCache sampler only. No SoloQ (its patch is outside this check), no trace.
"""
import collections
import sys
import time
import types

import torch

import generate as G


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


def _delta(ds, before):
    return [{p: d[p] - b.get(p, 0) for p in d} for d, b in zip(ds, before)]


class GraphStack:
    """Installs the graph stack on a model for one cell; `close()` takes it away."""

    def __init__(self, model, ctrl, sched, n_layers, gen_length, block_length, keep=4):
        if "forward" in vars(model):
            raise SystemExit("--graphs: the model already has a forward of its own (a patch); not covered")
        self.model, self.ctrl = model, ctrl
        self.gen, self.blk, self.keep = gen_length, block_length, keep
        self.dev = next(model.parameters()).device
        self._orig = model.forward          # the bound method of the class (the instance has no forward of its own)
        self._span = sys.modules[type(model.model).__module__]._replace_span
        # the active layers of each pass type, as the sampler arms them (_Depth.arm; a static schedule)
        dep = G._Depth(sched, ctrl, n_layers)
        dep.arm(G._state(1.0, 0, 1, 0, True, block_slice=(0, block_length)), force_full=True)
        self.act = dict(cache_write=ctrl._active)
        dep.arm(G._state(0.5, 0, 1, 1, False))
        self.act["refine"] = ctrl._active
        ctrl.arm(None)
        self.sets = collections.OrderedDict()
        self.cur = None
        self.capture_s, self.captures, self.graphs, self.evictions = 0.0, 0, 0, 0
        self.replays = dict(refine=0, cache_write=0)
        self.mem_max = 0
        model.forward = self.forward

    def _cw(self, ids):
        self.ctrl.arm(self.act["cache_write"])
        return self._orig(ids, use_cache=True)

    def _ref(self, ids, pkv, rp):
        self.ctrl.arm(self.act["refine"])
        return self._orig(ids, past_key_values=pkv, use_cache=True, replace_position=rp).logits

    def prepare(self, Lp):
        """The graphs of a problem with a prompt of Lp tokens (call before its timer)."""
        L = int(Lp) + self.gen
        if L in self.sets:
            self.sets.move_to_end(L)
            return
        while len(self.sets) >= self.keep:
            self.sets.popitem(last=False)
            self.evictions += 1
        t0 = time.perf_counter()
        self.sets[L] = self._capture(int(Lp), L)
        torch.cuda.synchronize()
        self.capture_s += time.perf_counter() - t0
        self.captures += 1
        self.mem_max = max(self.mem_max, torch.cuda.memory_reserved())

    @torch.no_grad()
    def _capture(self, Lp, L):
        c = self.ctrl
        st = (c.n_passes, c.layer_steps, c.n_skipped, len(c.history))
        cnt = _counters()
        snap = _snap(cnt)
        ids_full = torch.zeros((1, L), dtype=torch.long, device=self.dev)
        ids_blk = torch.zeros((1, self.blk), dtype=torch.long, device=self.dev)
        rps = {}
        for s in range(Lp, L, self.blk):
            rp = torch.zeros((1, L), dtype=torch.bool, device=self.dev)
            rp[:, s:s + self.blk] = True
            rps[s] = rp
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            # warm-up of each shape, eager; it also memoises the span of each mask (a host sync)
            pkv = self._cw(ids_full).past_key_values
            for s, rp in rps.items():
                assert self._span(rp) == (s, s + self.blk)
                self._ref(ids_blk, pkv, rp)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        del pkv
        pool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        before = _snap(cnt)
        with torch.cuda.graph(g, pool=pool):
            out = self._cw(ids_full)
        cw = (g, out, _delta(cnt, before))
        ref = {}
        for s, rp in rps.items():
            gr = torch.cuda.CUDAGraph()
            before = _snap(cnt)
            with torch.cuda.graph(gr, pool=pool):
                lo = self._ref(ids_blk, out.past_key_values, rp)
            ref[s] = (gr, lo, _delta(cnt, before))
        self.graphs += 1 + len(ref)
        # the capture is not part of the cell: the counts go back to their values before it
        _restore(cnt, snap)
        c.n_passes, c.layer_steps, c.n_skipped = st[:3]
        del c.history[st[3]:]
        c._pass_active = []
        c.arm(None)
        return dict(ids_full=ids_full, ids_blk=ids_blk, cw=cw, ref=ref, rps=rps)

    def _after(self, add):
        """The Python side of a replayed pass: the controller counts and the kernel-path counts."""
        c = self.ctrl
        for i in range(c.n_layers):
            c._tick(i, c._is_active(i))
        for d, a in zip(_counters(), add):
            for p, v in a.items():
                d[p] = d.get(p, 0) + v

    @torch.no_grad()
    def forward(self, input_ids=None, past_key_values=None, use_cache=False, replace_position=None, **kw):
        if kw or not use_cache or input_ids.shape[0] != 1:
            raise RuntimeError(f"--graphs: a pass that the graph stack does not cover ({sorted(kw)}, use_cache "
                               f"{use_cache}, {tuple(input_ids.shape)})")
        if past_key_values is None:
            kind, L = "cache_write", input_ids.shape[1]
            if L not in self.sets:
                raise RuntimeError(f"--graphs: no graphs for length {L} (prepare was not called)")
            gs = self.sets[L]
            g, out, add = gs["cw"]
            buf = gs["ids_full"]
        else:
            kind, gs = "refine", self.cur
            if gs is None or past_key_values is not gs["cw"][1].past_key_values:
                raise RuntimeError("--graphs: a refinement pass on a cache that is not that of the graphs")
            s, e = self._span(replace_position)
            if e - s != self.blk or s not in gs["ref"] or input_ids.shape[1] != self.blk:
                raise RuntimeError(f"--graphs: no refinement graph for the span ({s}, {e})")
            g, out, add = gs["ref"][s]
            buf = gs["ids_blk"]
        if self.ctrl._active != self.act[kind]:
            raise RuntimeError(f"--graphs: a {kind} pass with other active layers than at capture")
        buf.copy_(input_ids)
        g.replay()
        self.replays[kind] += 1
        self._after(add)
        if kind == "cache_write":
            self.cur = gs
            return out
        return types.SimpleNamespace(logits=out)

    def record(self):
        return dict(captures=self.captures, graphs=self.graphs, capture_s=self.capture_s, keep=self.keep,
                    evictions=self.evictions, mem_reserved_max=self.mem_max, replays=dict(self.replays))

    def close(self):
        del self.model.forward
        self.sets.clear()
        self.cur = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
