"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 8, 32  # N must be even -- each thread owns a 2x2 output tile


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(29)
    A = rng.uniform(-2.0, 2.0, size=(N, N))
    B = rng.uniform(-2.0, 2.0, size=(N, N))
    exact = A @ B
    denom = float(np.max(np.abs(exact)))

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()
    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"rel_err": float("inf"), "error": str(e)}

    gpu = GPU(3 * N * N)
    gpu.gmem[0:N * N] = A.flatten()
    gpu.gmem[N * N:2 * N * N] = B.flatten()
    gpu.gmem[2 * N * N:3 * N * N] = 0.0
    tiles = (N // 2) * (N // 2)
    params = {"A": 0, "B": N * N, "C": 2 * N * N, "N": N}

    try:
        grid = (tiles + BLOCK - 1) // BLOCK
        prog.launch(gpu, "coarsened_matmul", grid, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"rel_err": float("inf"), "error": str(e)}

    C = gpu.gmem[2 * N * N:3 * N * N].reshape(N, N)
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
