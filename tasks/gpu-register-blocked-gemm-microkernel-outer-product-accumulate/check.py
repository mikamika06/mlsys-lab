"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU). Compares its output
against a numpy A@B oracle, and compares the simulator's `transactions`
count against a FIXED naive one-thread-per-output baseline (embedded here,
never the learner's code) run through the exact same simulator.
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

# HARNESS baseline (fixed, not learner code): one thread per output
# element, no register blocking at all.
NAIVE_SRC = """
__global__ void gemm_regblock(float* C, const float* A, const float* B, int M, int N, int K) {
    int row = threadIdx.x / N;
    int col = threadIdx.x % N;
    float acc = 0.0f;
    int k = 0;
    while (k < K) {
        acc += A[row * K + k] * B[k * N + col];
        k = k + 1;
    }
    C[row * N + col] = acc;
}
"""


def _setup(rng):
    A = rng.uniform(-2.0, 2.0, size=(M, K))
    B = rng.uniform(-2.0, 2.0, size=(K, N))
    gpu = GPU(M * K + K * N + M * N)
    gpu.gmem[0:M * K] = A.reshape(-1)
    gpu.gmem[M * K:M * K + K * N] = B.reshape(-1)
    gpu.gmem[M * K + K * N:] = 0.0
    params = {"C": M * K + K * N, "A": 0, "B": M * K, "M": M, "N": N, "K": K}
    return A, B, gpu, params


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(31)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
        naive_prog = CudaProgram(NAIVE_SRC)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "size_ratio": 0.0, "error": str(e)}

    A, B, gpu, params = _setup(rng)
    try:
        m = prog.launch(gpu, "gemm_regblock", 1, (M // 2) * (N // 2), params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "size_ratio": 0.0, "error": str(e)}

    oracle = A @ B
    got = gpu.gmem[M * K + K * N:].reshape(M, N)
    max_err = float(np.max(np.abs(got - oracle)))

    _, _, gpu_naive, params_naive = _setup(rng)
    m_naive = naive_prog.launch(gpu_naive, "gemm_regblock", 1, M * N, params_naive)

    reg_tx = int(m["transactions"])
    naive_tx = int(m_naive["transactions"])
    size_ratio = float(naive_tx) / reg_tx if reg_tx > 0 else 0.0
    return {"max_abs_err": max_err, "size_ratio": size_ratio,
            "regblock_transactions": reg_tx, "naive_transactions": naive_tx}


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
