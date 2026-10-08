"""Phase 2 amendment A1, check 1: `patch_from_env(model)` with SOLOQ_WA and SOLOQ_KV unset changes nothing.

Family A (LLaDA) is not bit-reproducible from run to run, so this check compares the model object before and
after the call, not the decoding output (`results/phase2/PREREG.md`, section 10, A1):
- the SHA-256 of each parameter and each buffer;
- the class of each module;
- the number of forward hooks and forward pre-hooks on each module, and the global module hooks;
- the `forward` of the model class and the functions of the modeling module (a KV patch replaces them).

SoloQ is private. This file holds only the call. The code is imported from SOLOQ_PATH, outside the fork.

    SOLOQ_PATH=/path/to/SoloQ python phase2_noop_check.py --out <noop_A.json>
"""
import argparse
import hashlib
import json
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model.modeling_llada import LLaDAModelLM                          # noqa: E402
import model.modeling_llada as modeling                                # noqa: E402


def snapshot(model, tensors=True):
    """The state that the call must not change. `tensors=False` skips the hashes (a structure check only)."""
    hashed = {}
    for kind, it in ((("param", model.named_parameters()), ("buffer", model.named_buffers())) if tensors else ()):
        for name, t in it:
            b = t.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes()
            hashed[f"{kind}:{name}"] = (hashlib.sha256(b).hexdigest(), str(t.dtype), list(t.shape), str(t.device))
    classes = {name: f"{type(m).__module__}.{type(m).__qualname__}" for name, m in model.named_modules()}
    hooks = {name: (len(m._forward_hooks), len(m._forward_pre_hooks)) for name, m in model.named_modules()}
    g = torch.nn.modules.module
    glob = (len(g._global_forward_hooks), len(g._global_forward_pre_hooks))
    funcs = {k: id(v) for k, v in vars(modeling).items() if isinstance(v, (types.FunctionType, type))}
    return dict(tensors=hashed, classes=classes, hooks=hooks, global_hooks=glob,
                cls_forward=id(type(model).forward), funcs=funcs)


def diff(a, b):
    out = {}
    for key in ("tensors", "classes", "hooks", "funcs"):
        ka, kb = a[key], b[key]
        out[key] = dict(n_before=len(ka), n_after=len(kb),
                        changed=sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))[:50])
    out["global_hooks"] = dict(before=a["global_hooks"], after=b["global_hooks"])
    out["cls_forward_same"] = a["cls_forward"] == b["cls_forward"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = {k: v for k, v in os.environ.items() if k.startswith("SOLOQ_")}
    if "SOLOQ_PATH" not in env:
        raise SystemExit("SOLOQ_PATH is not set")
    if any(k in env for k in ("SOLOQ_WA", "SOLOQ_KV")):
        raise SystemExit(f"A1 needs SOLOQ_WA and SOLOQ_KV unset: {env}")
    model = LLaDAModelLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    before = snapshot(model)
    sys.path.insert(0, env["SOLOQ_PATH"])
    import soloq.bridge
    ret = soloq.bridge.patch_from_env(model)
    after = snapshot(model)
    d = diff(before, after)
    ok = (not any(d[k]["changed"] for k in ("tensors", "classes", "hooks", "funcs"))
          and d["global_hooks"]["before"] == d["global_hooks"]["after"] and d["cls_forward_same"] and ret is None)
    res = dict(check="A1.1 Family A no-op", model=a.model, soloq_env=env, patch_return=repr(ret),
               n_tensors=len(before["tensors"]), n_modules=len(before["classes"]),
               n_hooks_before=sum(x[0] + x[1] for x in before["hooks"].values()), diff=d, passed=ok)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"A1.1 Family A no-op: {'PASS' if ok else 'FAIL'}  tensors {res['n_tensors']}  modules {res['n_modules']}  "
          f"hooks {res['n_hooks_before']}  -> {a.out}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
