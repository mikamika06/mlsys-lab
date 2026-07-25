"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Checks a ragged-tile (M, N, K not multiples of the tile size) ragged GEMM
against a plain numpy A@B oracle. The scratch memory just past A/B/C's
real extent is deliberately filled with large nonzero "poison" values (not
zeros), so an unguarded out-of-bounds tile load reads visibly wrong data
instead of silently getting lucky with a zero.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M = N = K = 6
TILE = 4
PAD = 20
POISON = 777.0


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(5)
    A = rng.randn(M, K).astype(np.float64)
    B = rng.randn(K, N).astype(np.float64)
    ref = A @ B

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    c_base = 0
    a_base = M * N
    b_base = a_base + M * K
    tail = b_base + K * N
    gpu = GPU(tail + PAD, smem_size=2 * TILE * TILE)
    gpu.gmem[:] = POISON  # poison every byte first...
    gpu.gmem[c_base:c_base + M * N] = 0.0
    gpu.gmem[a_base:a_base + M * K] = A.ravel()
    gpu.gmem[b_base:b_base + K * N] = B.ravel()
    gpu.gmem[tail:tail + PAD] = POISON  # ...including the tail scratch region

    params = {"C": c_base, "A": a_base, "B": b_base, "M": M, "N": N, "K": K}
    try:
        prog.launch(gpu, "gemm_ragged_tile", 4, TILE * TILE, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    out = gpu.gmem[c_base:c_base + M * N].reshape(M, N)
    max_err = float(np.max(np.abs(out - ref)))
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
