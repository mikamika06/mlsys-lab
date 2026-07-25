"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the mixed-precision QK^T output against a numpy oracle that rounds each
input to the same exact fp16 grid before multiplying.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M = N = 4
D = 8

S_BASE = 0
Q_BASE = M * N
K_BASE = Q_BASE + M * D
GMEM_SIZE = K_BASE + N * D


def _round16(x):
    return np.floor(x * 1024.0 + 0.5) / 1024.0


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(9)
    Q = rng.uniform(1.0, 2.0, size=(M, D)).astype(np.float64)
    K = rng.uniform(1.0, 2.0, size=(N, D)).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[S_BASE:S_BASE + M * N] = -1.0
    gpu.gmem[Q_BASE:Q_BASE + M * D] = Q.reshape(-1)
    gpu.gmem[K_BASE:K_BASE + N * D] = K.reshape(-1)

    params = {"S": S_BASE, "Q": Q_BASE, "K": K_BASE, "M": M, "N": N, "D": D}
    try:
        prog.launch(gpu, "qkt_mixed_precision", 1, 16, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    Qh, Kh = _round16(Q), _round16(K)
    ref = (Qh @ Kh.T) / np.sqrt(D)
    Sout = gpu.gmem[S_BASE:S_BASE + M * N].reshape(M, N)
    max_err = float(np.max(np.abs(Sout - ref)))
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
