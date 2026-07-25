"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

A single-block, 2-K-tile matmul (M=N=8, K=16, tile_k=8). The reference C is
computed by an explicit Python loop that sums tile 0's contribution then
tile 1's, in the same order the kernel is expected to -- so a correct
kernel matches it EXACTLY, regardless of how it overlaps the tile-1
prefetch with the tile-0 compute. This simulator drives a block's threads
generator-by-generator between __syncthreads() calls (each thread runs its
whole segment before the next thread starts, not truly in lockstep), so
overlapping a prefetch with a compute by reusing the SAME shared buffer for
both tiles is not just slower than double buffering here -- it is WRONG:
early threads finish writing tile 1's data over shared memory before later
threads have read tile 0's data out of it for their own compute step.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M, N, K, TILE_K = 8, 8, 16, 8


def _fixture():
    rng = np.random.RandomState(2026)
    A = rng.uniform(-1.0, 1.0, size=(M, K))
    B = rng.uniform(-1.0, 1.0, size=(K, N))
    return A, B


def _reference(A, B):
    # Same order a correct kernel accumulates in: every element of tile 0
    # first, then every element of tile 1 -- so a correct kernel matches
    # this EXACTLY, not just approximately.
    C = np.zeros((M, N))
    for row in range(M):
        for col in range(N):
            acc = 0.0
            for k in range(TILE_K):
                acc = acc + A[row, k] * B[k, col]
            for k in range(TILE_K):
                acc = acc + A[row, TILE_K + k] * B[TILE_K + k, col]
            C[row, col] = acc
    return C


def grade(srcfile: str = "solve.cu") -> dict:
    A, B = _fixture()
    C_ref = _reference(A, B)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    a_size, b_size, c_size = M * K, K * N, M * N
    gpu = GPU(a_size + b_size + c_size, smem_size=300)
    gpu.gmem[0:a_size] = A.flatten()
    gpu.gmem[a_size:a_size + b_size] = B.flatten()
    gpu.gmem[a_size + b_size:a_size + b_size + c_size] = 0.0

    params = {"C": a_size + b_size, "A": 0, "B": a_size,
              "M": M, "N": N, "K": K, "tile_k": TILE_K}
    try:
        prog.launch(gpu, "tiled_matmul_double_buffered", 1, M * N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    C_got = gpu.gmem[a_size + b_size:a_size + b_size + c_size].reshape(M, N)
    max_err = float(np.max(np.abs(C_got - C_ref)))
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
