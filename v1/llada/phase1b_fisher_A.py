"""Phase 1b Part C (v) on Family A: the Fisher set of `famC/phase1b_C.py` on the Phase 1 bank (`results/phase1b/PREREG.md`
C). The bank is read from the Phase 1 state file, never rebuilt; the reference pass and the commit set are
`phase1_map.reference`, so every committed token lines up with `results/phase1/tok_A.npz`, whose single-layer skip
KL (configs L1..L30) is the other side of the bridge -- it is not recomputed here.

Per committed token, one forward with grad and three backwards with retained graph (y = the committed token,
F-emp; two samples y ~ p, F-true). Per layer and parameter group the sum of squared parameter gradients (post-
accumulate hooks free each gradient at once); per layer the activation Fisher on the layer's residual update
(grad of the block output == grad of delta_l, over the block's positions x dims), KL2 = 1/2 sum F-act * delta^2 and
the rank-1 form 1/2 (g . delta)^2 (true samples).

    python phase1b_fisher_A.py --bank-state /scratch1/.../map_A.state.pt --tok results/phase1/tok_A.npz \
        --out results/phase1b/fisher_A
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import phase1_map as M                                                  # noqa: E402
from model.modeling_llada import LLaDAModelLM                           # noqa: E402

GROUPS = ["q", "k", "v", "o", "ff_gate", "ff_up", "ff_down", "attn_norm", "ff_norm"]
NAMES = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "attn_out": "o", "ff_proj": "ff_gate", "up_proj": "ff_up",
         "ff_out": "ff_down", "attn_norm": "attn_norm", "ff_norm": "ff_norm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--bank-state", required=True)
    ap.add_argument("--tok", required=True, help="results/phase1/tok_A.npz (canvas order and commit counts)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--n-true", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    dev = torch.device("cuda")
    # the model's compiled kernels donate their saved buffers to the backward, which forbids the retained graph the
    # three backwards per token need; turning donation off changes memory use only, not values
    torch._functorch.config.donated_buffer = False
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    model.requires_grad_(False)
    blocks = model.model.transformer.blocks
    L = len(blocks)
    bank = torch.load(a.bank_state, weights_only=False)["bank"]
    tk = np.load(a.tok)
    order = tk["c_bank_idx"][:a.limit] if a.limit else tk["c_bank_idx"]
    n_commit_tok = tk["c_n_commit"]

    gi = {g: i for i, g in enumerate(GROUPS)}
    acc = torch.zeros(L, len(GROUPS), device=dev, dtype=torch.float64)
    numel = np.zeros((L, len(GROUPS)), np.int64)
    for l, blk in enumerate(blocks):
        for name, p in blk.named_parameters():
            grp = NAMES[name.split(".")[0]]
            p.requires_grad_(True)
            numel[l, gi[grp]] += p.numel()

            def hook(param, l=l, j=gi[grp]):
                acc[l, j] += param.grad.detach().double().pow(2).sum()
                param.grad = None
            p.register_post_accumulate_grad_hook(hook)

    cap = {"on": False}

    def pre(l):
        def h(mod, args, kwargs):
            if cap["on"]:
                cap["in"][l] = (args[0] if args else kwargs["x"]).detach()
        return h

    def post(l):
        def h(mod, args, kwargs, out):
            if cap["on"]:
                x = out[0]
                x.retain_grad()
                cap["h"][l] = x
                cap["d"][l] = x.detach().float() - cap["in"][l].float()
        return h
    for l, blk in enumerate(blocks):
        blk.register_forward_pre_hook(pre(l), with_kwargs=True)
        blk.register_forward_hook(post(l), with_kwargs=True)

    state_path = a.state or (a.out + ".state.pt")
    st = torch.load(state_path, weights_only=False) if os.path.exists(state_path) else dict(done=0, rows=[])
    gen = torch.Generator(device=dev)
    t0 = time.perf_counter()
    mism = 0
    for n in range(st["done"], len(order)):
        c = bank[int(order[n])]
        with torch.no_grad():
            pkv = model(c["x"], use_cache=True).past_key_values
        # the block pass writes its K/V into the cache in place; tensors made under no_grad cannot take a
        # grad-tracked in-place write, so the cache is re-materialised (values unchanged) with grad mode on
        pkv = [tuple(t.detach().clone() for t in layer) for layer in pkv]
        cap.update(on=True, h={}, d={}, **{"in": {}})
        with torch.enable_grad():
            lg = model(c["x"][:, c["s"]:c["e"]], past_key_values=pkv, use_cache=True,
                       replace_position=c["rp"]).logits
        cap["on"] = False
        mb = (c["x"][:, c["s"]:c["e"]] == M.MASK_ID)
        with torch.no_grad():
            _x0, ti, _conf = M.G.get_transfer_index(lg.detach(), 0.0, "low_confidence", mb, c["x"][:, c["s"]:c["e"]],
                                                    None, a.threshold, return_confidence=True)
        pos = ti[0].nonzero().squeeze(-1).tolist()
        if len(pos) != int(n_commit_tok[n]):
            mism += 1
        lp = torch.log_softmax(lg[0].float(), -1)
        T = len(pos)
        Femp = np.zeros((T, L, len(GROUPS))); Ftrue = np.zeros_like(Femp)
        Aemp = np.zeros((T, L)); Atrue = np.zeros((T, L)); K2emp = np.zeros((T, L)); K2true = np.zeros((T, L))
        Q1true = np.zeros((T, L))
        d2 = {l: cap["d"][l].pow(2) for l in range(L)}
        gen.manual_seed(int(order[n]))
        for t_i, j in enumerate(pos):
            probs = lp[j].detach().exp()
            targets = [("emp", int(probs.argmax()))] + [("true", int(torch.multinomial(probs, 1, generator=gen)))
                                                        for _ in range(a.n_true)]
            for kind, y in targets:
                for l in range(L):
                    cap["h"][l].grad = None
                acc.zero_()
                lp[j, y].backward(retain_graph=True)
                with torch.no_grad():
                    gs = [cap["h"][l].grad.float() for l in range(L)]
                    v0 = acc.cpu().numpy()
                    v1 = torch.stack([g.pow(2).sum() for g in gs]).double().cpu().numpy()
                    v2 = torch.stack([0.5 * (g.pow(2) * d2[l]).sum() for l, g in enumerate(gs)]).double().cpu().numpy()
                    v3 = torch.stack([0.5 * (g * cap["d"][l]).sum().pow(2) for l, g in enumerate(gs)]).double().cpu().numpy()
                if kind == "emp":
                    Femp[t_i] = v0; Aemp[t_i] = v1; K2emp[t_i] = v2
                else:
                    Ftrue[t_i] += v0 / a.n_true; Atrue[t_i] += v1 / a.n_true
                    K2true[t_i] += v2 / a.n_true; Q1true[t_i] += v3 / a.n_true
        cap.update(h={}, d={}, **{"in": {}})
        del lg, lp, pkv
        st["rows"].append(dict(n=n, bank_idx=int(order[n]), pos=np.array(pos), F_emp=Femp.astype(np.float32),
                               F_true=Ftrue.astype(np.float32), Fact_emp=Aemp.astype(np.float32),
                               Fact_true=Atrue.astype(np.float32), KL2_emp=K2emp.astype(np.float32),
                               KL2_true=K2true.astype(np.float32), KL2q_true=Q1true.astype(np.float32)))
        st["done"] = n + 1
        if (n + 1) % 200 == 0 or n + 1 == len(order):
            print(f"  fisher A: {n+1}/{len(order)} ({time.perf_counter()-t0:.0f}s, commit-count mismatches vs tok_A "
                  f"{mism} this allocation)", flush=True)
            torch.save(st, state_path + ".tmp"); os.replace(state_path + ".tmp", state_path)
    rows = st["rows"]
    cat = lambda k: np.concatenate([r[k] for r in rows])                                # noqa: E731
    np.savez_compressed(a.out + ".npz", tok_n=np.concatenate([np.full(len(r["pos"]), r["n"], np.int32) for r in rows]),
                        tok_pos=cat("pos"), F_emp=cat("F_emp"), F_true=cat("F_true"), Fact_emp=cat("Fact_emp"),
                        Fact_true=cat("Fact_true"), KL2_emp=cat("KL2_emp"), KL2_true=cat("KL2_true"),
                        KL2q_true=cat("KL2q_true"), c_n_commit=np.array([len(r["pos"]) for r in rows]),
                        numel=numel, groups=np.array(GROUPS))
    json.dump(dict(model=a.model, gpu=torch.cuda.get_device_name(0), args=vars(a), groups=GROUPS, L=L,
                   n_canvases=len(rows), note="tok_n indexes tok_A's canvas axis (c_bank_idx order)"),
              open(a.out + ".json", "w"), indent=1)
    print(f"written {a.out}.npz")


if __name__ == "__main__":
    main()
