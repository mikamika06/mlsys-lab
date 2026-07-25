"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
S=8 query/key/value rows, d=4 head dim, one thread per query row.
Compares the attention output against a numpy softmax(QK^T/sqrt(d))V
oracle.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
S, D, BK = 8, 4, 4


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(3)
    Q = rng.randn(S, D).astype(np.float64) * 0.5
    K = rng.randn(S, D).astype(np.float64) * 0.5
    V = rng.randn(S, D).astype(np.float64) * 0.5

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(4 * S * D)
    gpu.gmem[0 * S * D:1 * S * D] = 0.0            # out = gmem[0 : S*D]
    gpu.gmem[1 * S * D:2 * S * D] = Q.reshape(-1)   # Q   = gmem[S*D : 2*S*D]
    gpu.gmem[2 * S * D:3 * S * D] = K.reshape(-1)   # K   = gmem[2*S*D : 3*S*D]
    gpu.gmem[3 * S * D:4 * S * D] = V.reshape(-1)   # V   = gmem[3*S*D : 4*S*D]

    params = {
        "out": 0 * S * D, "Q": 1 * S * D, "K": 2 * S * D, "V": 3 * S * D,
        "S": S, "d": D, "bk": BK,
    }
    try:
        prog.launch(gpu, "flash_attn", 1, S, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    scores = (Q @ K.T) / np.sqrt(D)
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=1, keepdims=True)
    ref_out = (weights @ V).reshape(-1)

    got_out = gpu.gmem[0:S * D]
    max_err = float(np.max(np.abs(got_out - ref_out)))
    return {"max_abs_err": max_err}


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
