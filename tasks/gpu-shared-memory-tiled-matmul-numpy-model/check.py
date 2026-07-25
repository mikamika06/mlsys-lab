"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the tiled-matmul output against a numpy A@B oracle, and reports the
simulator's measured transaction count (the shared-memory tiling should
keep it low -- every A/B element is loaded from global memory once per
K-sub-tile and reused TILE times out of shared memory, instead of being
re-read from global memory for every output element that needs it).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M = N = K = 8
TILE = 4
GRID, BLOCK = (N // TILE) * (M // TILE), TILE * TILE

C_BASE = 0
A_BASE = M * N
B_BASE = A_BASE + M * K
GMEM_SIZE = B_BASE + K * N


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(3)
    A = rng.uniform(-1, 1, size=(M, K)).astype(np.float64)
    B = rng.uniform(-1, 1, size=(K, N)).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    gpu = GPU(GMEM_SIZE, smem_size=32)
    gpu.gmem[C_BASE:C_BASE + M * N] = -1.0
    gpu.gmem[A_BASE:A_BASE + M * K] = A.reshape(-1)
    gpu.gmem[B_BASE:B_BASE + K * N] = B.reshape(-1)

    params = {"C": C_BASE, "A": A_BASE, "B": B_BASE, "M": M, "N": N, "K": K}
    try:
        m = prog.launch(gpu, "tiled_matmul", GRID, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    ref = A @ B
    Cout = gpu.gmem[C_BASE:C_BASE + M * N].reshape(M, N)
    max_err = float(np.max(np.abs(Cout - ref)))
    return {"max_abs_err": max_err, "transactions": int(m["transactions"])}


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
