"""Grade a REAL CUDA-C tiled matmul: parse solve.cu with arena.cuda_c.CudaProgram
and execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Correctness is checked against a numpy oracle; transaction_ratio against an
untiled baseline the grader measures ITSELF (a plain Python 2D kernel run
directly on the simulator — not something the learner submits, just the
grader's own internal reference point, exactly as the 2D `cuda_sim` API
already supports, independent of the 1D-only CUDA-C frontend).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 32
TILE = 16
TILES_PER_ROW = N // TILE


def _layout(rng):
    """A, B and C laid out back to back in one flat global memory."""
    A = rng.random((N, N))
    B = rng.random((N, N))
    gmem = np.concatenate([A.ravel(), B.ravel(), np.zeros(N * N)])
    return A, B, gmem


def _naive(t, n, off_b, off_c):
    """Untiled baseline (plain-Python 2D kernel) — the grader's own internal
    reference point for transaction_ratio, never what the learner submits."""
    row = t.blockIdx.y * t.blockDim.y + t.threadIdx.y
    col = t.blockIdx.x * t.blockDim.x + t.threadIdx.x
    if row >= n or col >= n:
        return
    acc = 0.0
    for k in range(n):
        acc += t.gload(row * n + k) * t.gload(off_b + k * n + col)
        t.alu(2)
    t.gstore(off_c + row * n + col, acc)


def _baseline_transactions(gmem):
    g = GPU(len(gmem), smem_size=1)
    g.gmem[:] = gmem
    m = g.launch(_naive, (TILES_PER_ROW, TILES_PER_ROW), (TILE, TILE), N, N * N, 2 * N * N)
    return max(float(m["transactions"]), 1.0)


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.default_rng(0)
    A, B, gmem = _layout(rng)
    ref = A @ B  # numpy oracle

    base_tx = _baseline_transactions(gmem)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "transaction_ratio": float("inf"), "error": str(e)}

    gpu = GPU(3 * N * N, smem_size=2 * TILE * TILE)
    gpu.gmem[:] = gmem

    params = {"a": 0, "b": N * N, "c": 2 * N * N, "n": N, "tiles_per_row": TILES_PER_ROW}
    try:
        m = prog.launch(gpu, "tiled_matmul", TILES_PER_ROW * TILES_PER_ROW, TILE * TILE, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "transaction_ratio": float("inf"), "error": str(e)}

    out = gpu.gmem[2 * N * N:].reshape(N, N)
    max_err = float(np.max(np.abs(out - ref)))
    return {"max_abs_err": max_err, "transaction_ratio": float(m["transactions"]) / base_tx}


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
