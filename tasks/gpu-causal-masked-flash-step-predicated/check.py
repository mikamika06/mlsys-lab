"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the resulting O against a numpy causal-softmax-attention oracle, and reports
the warp divergence count the simulator observed (predication vs an early
skip on `j > i` is the whole point).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
S = 32          # queries = keys = one warp
D = 4           # head dim


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.default_rng(0)
    q = rng.standard_normal((S, D))
    k = rng.standard_normal((S, D))
    v = rng.standard_normal((S, D))
    scale = 1.0 / np.sqrt(D)

    # oracle: causal softmax attention, computed here with numpy
    logits = (q @ k.T) * scale
    mask = np.tril(np.ones((S, S), dtype=bool))
    logits = np.where(mask, logits, -np.inf)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    ref = p @ v

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    off_k, off_v, off_o = S * D, 2 * S * D, 3 * S * D
    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "divergences": 1e9, "error": str(e)}

    gpu = GPU(4 * S * D)
    gpu.gmem[0:S * D] = q.ravel()
    gpu.gmem[off_k:off_k + S * D] = k.ravel()
    gpu.gmem[off_v:off_v + S * D] = v.ravel()
    gpu.gmem[off_o:off_o + S * D] = 0.0

    params = {"q": 0, "k": off_k, "v": off_v, "o": off_o, "s": S, "d": D, "scale": float(scale)}
    try:
        m = prog.launch(gpu, "flash_step", 1, S, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "divergences": 1e9, "error": str(e)}

    out = gpu.gmem[off_o:off_o + S * D].reshape(S, D)
    max_err = float(np.max(np.abs(out - ref)))
    return {"max_abs_err": max_err, "divergences": float(m["divergences"])}


if __name__ == "__main__":
    srcfile = sys.argv[1] if len(sys.argv) > 1 else "solve.cu"
    metrics = grade(srcfile)
    meta = json.load(open(os.path.join(HERE, "meta.json")))
    ok = True
    for g in meta["gates"]:
        v = metrics.get(g["metric"])
        gate_ok = v is not None and (
            v <= g["threshold"] if g["op"] == "<=" else
            v >= g["threshold"] if g["op"] == ">=" else v == g["threshold"]
        )
        ok = ok and gate_ok
        print(f"{'PASS' if gate_ok else 'FAIL'}  {g['metric']}={v}  ({g['op']} {g['threshold']})")
    print(json.dumps(metrics, indent=2))
    sys.exit(0 if ok else 1)
