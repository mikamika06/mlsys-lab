"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the masked block-matmul output against a numpy A@B oracle on a
matrix size that does NOT evenly divide the block's tile size, so the
boundary tiles' overhanging threads must be masked correctly.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M = N = K = 5
BLOCK_TILE = 4
GRID, BLOCK = 4, 16  # ceil(5/4)^2 = 4 blocks of 16 threads each

C_BASE = 0
A_BASE = M * N
B_BASE = A_BASE + M * K
GMEM_SIZE = B_BASE + K * N


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(4)
    A = rng.uniform(-1, 1, size=(M, K)).astype(np.float64)
    B = rng.uniform(-1, 1, size=(K, N)).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[C_BASE:C_BASE + M * N] = -999.0
    gpu.gmem[A_BASE:A_BASE + M * K] = A.reshape(-1)
    gpu.gmem[B_BASE:B_BASE + K * N] = B.reshape(-1)

    params = {"C": C_BASE, "A": A_BASE, "B": B_BASE, "M": M, "N": N, "K": K}
    try:
        prog.launch(gpu, "block_matmul_masked", GRID, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref = A @ B
    Cout = gpu.gmem[C_BASE:C_BASE + M * N].reshape(M, N)
    max_err = float(np.max(np.abs(Cout - ref)))
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
