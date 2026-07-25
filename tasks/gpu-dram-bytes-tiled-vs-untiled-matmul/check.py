"""Grade a REAL CUDA-C solve.cu (a `matmul_tiled` kernel): parse it with
arena.cuda_c.CudaProgram and execute it on the software GPU. Compares its
output against a numpy oracle, and compares its `transactions` (global-
memory DRAM-traffic proxy) against a FIXED naive matmul baseline -- an
always-correct kernel embedded here, run through the exact same simulator,
never the learner's code -- to grade real, measured DRAM-traffic reuse.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M = N = K = 32
T = 16

# HARNESS baseline (fixed, not learner code): every thread re-reads its
# own K-length row of A and column of B straight from global memory,
# with zero reuse across threads. 1D grid/block (this simulator's
# CUDA-C frontend is 1D-only), tile coordinates derived by hand.
NAIVE_SRC = """
__global__ void matmul_naive(float* C, const float* A, const float* B, int M, int N, int K) {
    int lane = threadIdx.x;
    int tx = lane % 16;
    int ty = lane / 16;
    int blocksPerRow = N / 16;
    int blockCol = blockIdx.x % blocksPerRow;
    int blockRow = blockIdx.x / blocksPerRow;
    int row = blockRow * 16 + ty;
    int col = blockCol * 16 + tx;
    float acc = 0.0f;
    int k = 0;
    while (k < K) {
        acc += A[row * K + k] * B[k * N + col];
        k = k + 1;
    }
    C[row * N + col] = acc;
}
"""


def _matmul_setup(rng):
    A = rng.uniform(-2.0, 2.0, size=(M, K))
    B = rng.uniform(-2.0, 2.0, size=(K, N))
    gpu = GPU(M * K + K * N + M * N, smem_size=512)
    gpu.gmem[0:M * K] = A.reshape(-1)
    gpu.gmem[M * K:M * K + K * N] = B.reshape(-1)
    gpu.gmem[M * K + K * N:] = 0.0
    params = {"C": M * K + K * N, "A": 0, "B": M * K, "M": M, "N": N, "K": K}
    return A, B, gpu, params


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(5)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
        naive_prog = CudaProgram(NAIVE_SRC)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "size_ratio": 0.0, "error": str(e)}

    grid = (M // T) * (N // T)
    block = T * T

    A, B, gpu, params = _matmul_setup(rng)
    try:
        m = prog.launch(gpu, "matmul_tiled", grid, block, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "size_ratio": 0.0, "error": str(e)}

    oracle = A @ B
    got = gpu.gmem[M * K + K * N:].reshape(M, N)
    max_err = float(np.max(np.abs(got - oracle)))

    _, _, gpu_naive, params_naive = _matmul_setup(rng)
    m_naive = naive_prog.launch(gpu_naive, "matmul_naive", grid, block, params_naive)

    tiled_tx = int(m["transactions"])
    naive_tx = int(m_naive["transactions"])
    size_ratio = float(naive_tx) / tiled_tx if tiled_tx > 0 else 0.0
    return {"max_abs_err": max_err, "size_ratio": size_ratio,
            "tiled_transactions": tiled_tx, "naive_transactions": naive_tx}


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
