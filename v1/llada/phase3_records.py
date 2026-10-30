"""Phase 3: the records and checks of each cell, shared by both runners (results/phase3/PREREG.md, amendments A13,
change 11; A14, changes 12 and 13; A15, changes 4 and 6).

- run_record(): the fork commit and `git status --porcelain` of the worktree that runs, the Slurm job, the restart,
  the host and the GPU (name and UUID). The runner writes it into each summary.json with the start and end time of
  the cell (now()).
- snapshot(): the folder of the Hub cache that the job reads (A14, change 5).
- sha256(): the sha256 of a file (a compressed-weights artifact), computed once per process.
- soloq_state(): SoloQ is private, so the fork names no SoloQ variable. With SOLOQ_PATH set, the code is imported from
  there and its own config_from_env() says if a SoloQ configuration is active. A cell that is not a SoloQ cell stops
  if one is active (A15, change 4).
- same_args(): a finished cell is skipped only if it was made with the same arguments and fork commit (A14,
  change 12); with one_allocation, also in the same job and restart.
- PassTimer and pass_stats(): the time, the count and the calls of each kernel path (path_calls(): Marlin or its bf16
  copy, the int4 kernel of AWQ or its bf16 copy) of each pass type (A15, change 3).
- The diagnosis of A13, change 4 and A15, change 7 (no reading): proc_stats() (memory of the GPU and of the process,
  the counts of the garbage collector), linear_call_ms() (one layer call at 32 and 1,024 rows), PassTracer (a
  torch.profiler trace of one refinement pass) and sparse_info() (the class of the 2:4 weight and the versions).
"""
import datetime
import gc
import hashlib
import os
import socket
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_SHA = {}

# The arguments that define a cell. A finished cell with other values of these is not this cell.
CELL_ARGS = ("model", "model_revision", "task", "split", "ids", "n_shot", "shots_split", "gen_length", "steps",
             "block_length", "small_block_size", "max_new_tokens", "threshold", "tau_r", "keep_last", "cells",
             "kl_json", "limit", "shard", "batch", "sampler", "prompt_style", "compress", "compress_dir", "gptq",
             "gptq_kernel", "gptq_offset", "marlin_row_max", "rtn_bits", "eb_gamma", "fixed_steps", "dus_base",
             "gsm8k_rule", "warmup", "deterministic", "deterministic_strict", "kernel_row_max", "regimes", "calib",
             "r_star", "soloq", "check_a7", "exact_prompts")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _git(*cmd, cwd=HERE):
    r = subprocess.run(["git", "-C", cwd, *cmd], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def gpu_uuid(idx=0):
    u = getattr(torch.cuda.get_device_properties(idx), "uuid", None)
    if u is not None:
        s = str(u)
        return s if s.startswith("GPU-") else f"GPU-{s}"
    r = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.split()[idx] if r.returncode == 0 and r.stdout.split() else None


def run_record():
    """The fork, the job and the GPU of this process."""
    return dict(fork_commit=_git("rev-parse", "HEAD"), fork_status=_git("status", "--porcelain"),
                worktree=_git("rev-parse", "--show-toplevel"), job=os.environ.get("SLURM_JOB_ID"),
                restart=int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0), host=socket.gethostname(),
                partition=os.environ.get("SLURM_JOB_PARTITION"), slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS"),
                gpu=torch.cuda.get_device_name(0), gpu_uuid=gpu_uuid(0), torch=torch.__version__,
                cuda=torch.version.cuda, python=sys.version.split()[0], start=now())


def snapshot(repo, revision=None, repo_type="model"):
    """The folder of the Hub cache that a job reads for this repository and revision (A14, change 5)."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo, revision=revision, repo_type=repo_type, local_files_only=True)
    except Exception as e:                  # noqa: BLE001 -- a record, never a stop
        return f"unknown ({type(e).__name__}: {e})"[:300]


def sha256(path):
    """The sha256 of a file, once per process (an artifact of several GB takes about 10 s)."""
    if path not in _SHA:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        _SHA[path] = h.hexdigest()
    return _SHA[path]


def soloq_state(model, soloq_cell):
    """Import SoloQ from SOLOQ_PATH (if set) and apply its patch. Returns the record of the cell.

    soloq_cell: the job asks for a SoloQ cell. Then SOLOQ_PATH must be set and config_from_env() must give a
    configuration. Else SoloQ must give none, or the cell stops."""
    path = os.environ.get("SOLOQ_PATH")
    rec = dict(soloq_cell=bool(soloq_cell), soloq_path_set=bool(path), patch_ran=False, config=None, commit=None)
    if not path:
        if soloq_cell:
            raise SystemExit("a SoloQ cell needs SOLOQ_PATH")
        return rec
    sys.path.insert(0, path)
    import soloq.bridge
    cfg = soloq.bridge.config_from_env()
    rec.update(config=cfg, commit=_git("rev-parse", "HEAD", cwd=path))
    if not soloq_cell:
        if cfg is not None:
            raise SystemExit(f"this is not a SoloQ cell, but SoloQ gives a configuration: {cfg}")
        return rec
    if cfg is None:
        raise SystemExit("a SoloQ cell, but SoloQ gives no configuration")
    soloq.bridge.patch_from_env(model)
    rec["patch_ran"] = True
    return rec


def same_args(stored, current, one_allocation=False):
    """Why a finished cell is not this cell (an empty list if it is)."""
    sa, ca = stored.get("args", {}), current["args"]
    bad = [k for k in CELL_ARGS if k in sa and k in ca and sa[k] != ca[k]]
    bad += [f"{k} (only in this run)" for k in CELL_ARGS if k in ca and k not in sa and ca[k] not in (None, False)]
    sr, cr = stored.get("run") or {}, current["run"]
    if sr.get("fork_commit") and sr["fork_commit"] != cr["fork_commit"]:
        bad.append(f"fork commit {sr['fork_commit']} != {cr['fork_commit']}")
    if one_allocation and (sr.get("job"), sr.get("restart")) != (cr["job"], cr["restart"]):
        bad.append(f"allocation {sr.get('job')}/{sr.get('restart')} != {cr['job']}/{cr['restart']}")
    return bad


# ------------------------------------------------------------------ the times and kernel paths of the pass types
def path_calls():
    """The calls of each path of the 4-bit kernels so far (host integers, no sync). Marlin (GPTQ): the kernel or the
    bf16 copy; the tinygemm int4 kernel (AWQ, phase3_compress.KernelLinear): the kernel or the bf16 copy."""
    out = {}
    gq = sys.modules.get("gptq_dequant")
    if gq is not None and hasattr(gq, "MarlinLinear"):
        out.update(marlin=gq.MarlinLinear.CALLS["marlin"], marlin_bf16=gq.MarlinLinear.CALLS["bf16"])
    pc = sys.modules.get("phase3_compress")
    if pc is not None:
        out.update(int4=pc.KernelLinear.CALLS["kernel"], int4_bf16=pc.KernelLinear.CALLS["bf16"])
    return out


class PassTimer(list):
    """The `timer` list of a sampler: each mark (kind, CUDA event) also keeps path_calls() at that mark."""

    def append(self, item):
        super().append((*item, path_calls()))


def pass_stats(timer):
    """Seconds, counts and kernel-path calls of each pass type of one timer list (marks from the start of each pass to
    the next mark). Call after a sync."""
    t, n, c = dict(cache_write=0.0, refine=0.0), dict(cache_write=0, refine=0), {}
    for a, b in zip(timer, timer[1:]):
        k = a[0]
        t[k] = t.get(k, 0.0) + a[1].elapsed_time(b[1]) / 1000.0
        n[k] = n.get(k, 0) + 1
        if len(a) > 2 and len(b) > 2:
            d = c.setdefault(k, {})
            for p, v in b[2].items():
                d[p] = d.get(p, 0) + v - a[2].get(p, 0)
    return t, n, c


def add_calls(tot, calls):
    """Add the kernel-path calls of one problem (or batch) to the totals of a cell."""
    for k, d in calls.items():
        for p, v in d.items():
            tot.setdefault(k, {})[p] = tot.setdefault(k, {}).get(p, 0) + v


# ------------------------------------------------------------------ the diagnosis of the 2:4 cells (no reading)
def proc_stats():
    """After a problem (outside the clock): the memory of the GPU and of the process, and the garbage collector."""
    with open("/proc/self/statm") as f:
        rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return dict(mem_alloc=torch.cuda.memory_allocated(), mem_reserved=torch.cuda.memory_reserved(),
                max_mem_alloc=torch.cuda.max_memory_allocated(), rss=rss, gc_count=list(gc.get_count()),
                gc_collections=[s["collections"] for s in gc.get_stats()])


@torch.no_grad()
def linear_call_ms(lin, rows=(32, 1024), reps=50, warm=5):
    """The median time (ms) of one call of a linear at each number of rows, with CUDA events."""
    dev = next(lin.buffers(), None)
    dev = dev.device if dev is not None else next(lin.parameters()).device
    out = {}
    for r in rows:
        x = torch.randn(1, r, lin.in_features, device=dev, dtype=torch.bfloat16)
        for _ in range(warm):
            lin(x)
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); lin(x); b.record(); b.synchronize()
            ts.append(a.elapsed_time(b))
        out[str(r)] = sorted(ts)[len(ts) // 2]
    return out


def sparse_info(lin):
    """The class of the weight of a 2:4 linear, and the versions of torch, CUDA and cuSPARSELt."""
    w = None
    for name in ("w_sparse", "weight", "w"):
        w = getattr(lin, name, None)
        if w is not None:
            break
    cus = getattr(torch.backends, "cusparselt", None)
    from torch.sparse import SparseSemiStructuredTensor as S
    return dict(module=type(lin).__name__, weight_class=None if w is None else type(w).__name__,
                torch=torch.__version__, cuda=torch.version.cuda,
                cusparselt=cus.version() if cus is not None and cus.is_available() else None,
                force_cutlass=bool(getattr(S, "_FORCE_CUTLASS", None)))


class PassTracer(PassTimer):
    """A `timer` list for the sampler that also traces passes with torch.profiler: for each pass type in `kinds`, the
    pass number `target` of that type (1 = the first of the problem). It starts the profiler at the mark of that pass
    and stops it at the next mark, with a sync (so the time of this problem is not clean; the runner marks it). With
    one type the trace goes to `path`; with more, to `path` with "_<type>" before ".json"."""

    def __init__(self, target, path, kinds=("refine",)):
        super().__init__()
        self.target, self.kinds, self.prof, self.done = target, tuple(kinds), None, False
        self.count = {k: 0 for k in self.kinds}
        self.paths = {k: path if len(self.kinds) == 1 else path[:-len(".json")] + f"_{k}.json" for k in self.kinds}
        self.taken, self.cur = set(), None

    def append(self, item):
        if self.prof is not None:
            torch.cuda.synchronize()
            self.prof.stop()
            self.prof.export_chrome_trace(self.paths[self.cur])
            self.prof = None
            self.taken.add(self.cur)
            self.done = True
        super().append(item)
        k = item[0]
        if k in self.count and k not in self.taken:
            self.count[k] += 1
            if self.count[k] == self.target:
                from torch.profiler import profile, ProfilerActivity
                self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
                self.prof.start()
                self.cur = k


def gpu_share(trace_path):
    """From a chrome trace of one pass: the sum of the kernel times over the time of the pass (A14, change 13)."""
    import json
    ev = json.load(open(trace_path))["traceEvents"]
    k = [e for e in ev if e.get("cat") == "kernel" and "dur" in e]
    if not k:
        return None
    t0 = min(e["ts"] for e in ev if "ts" in e and e.get("ph") == "X")
    t1 = max(e["ts"] + e.get("dur", 0) for e in ev if "ts" in e and e.get("ph") == "X")
    return dict(kernels=len(k), kernel_us=sum(e["dur"] for e in k), span_us=t1 - t0,
                share=sum(e["dur"] for e in k) / (t1 - t0) if t1 > t0 else None)
