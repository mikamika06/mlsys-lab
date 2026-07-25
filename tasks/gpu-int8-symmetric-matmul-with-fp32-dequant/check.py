"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Compares the dequant-fused int8 matmul against an exact float64 A @ W.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M, N, K = 8, 8, 16
BLOCK = 32


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(13)
    A = rng.uniform(-2.0, 2.0, size=(M, K))
    W = rng.uniform(-1.0, 1.0, size=(K, N))
    scale = float(np.max(np.abs(W)) / 127.0)
    exact = A @ W
    denom = float(np.max(np.abs(exact)))

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()
    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"rel_err": float("inf"), "error": str(e)}

    gpu = GPU(M * K + K * N + M * N)
    gpu.gmem[0:M * K] = A.flatten()
    gpu.gmem[M * K:M * K + K * N] = W.flatten()
    gpu.gmem[M * K + K * N:M * K + K * N + M * N] = 0.0
    params = {"A": 0, "W": M * K, "C": M * K + K * N, "M": M, "N": N, "K": K, "scale": scale}

    try:
        grid = (M * N + BLOCK - 1) // BLOCK
        prog.launch(gpu, "int8_dequant_matmul", grid, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"rel_err": float("inf"), "error": str(e)}

    C = gpu.gmem[M * K + K * N:M * K + K * N + M * N].reshape(M, N)
    rel_err = float(np.max(np.abs(C - exact))) / denom
    return {"rel_err": rel_err}


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
