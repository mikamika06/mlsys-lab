"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Checks correctness against a numpy GEMM oracle AND that the shared-memory
bank-conflict traffic matches a conflict-free lower bound the grader
computes itself (never hardcoded) by running its own padded,
double-buffered tile kernel through the same simulator, via the native
Thread API (not compiled from a .cu file).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TILE = 32
M = N = 32
K = 64
STRIDE = TILE + 1  # +1 padding


def _conflict_free_reference(t, A, B, C, K_):
    """The grader's own padded, double-buffered tiled GEMM, written
    directly against the simulator's native Thread API -- only used to
    measure this run's conflict-free LOWER BOUND on shared-memory
    traffic, independent of whatever solve.cu/ref.cu do."""
    base_a = t.salloc(2 * TILE * STRIDE)
    base_b = t.salloc(2 * TILE * STRIDE)
    col = t.threadIdx.x // TILE
    row = t.threadIdx.x % TILE

    t.sstore(base_a + row * STRIDE + col, t.gload(A + row * K_ + col))
    t.sstore(base_b + row * STRIDE + col, t.gload(B + row * N + col))
    yield

    acc = 0.0
    num_k_tiles = K_ // TILE
    for kt in range(num_k_tiles):
        buf = (kt % 2) * TILE * STRIDE
        nbuf = ((kt + 1) % 2) * TILE * STRIDE
        if kt + 1 < num_k_tiles:
            t.sstore(base_a + nbuf + row * STRIDE + col, t.gload(A + row * K_ + (kt + 1) * TILE + col))
            t.sstore(base_b + nbuf + row * STRIDE + col, t.gload(B + ((kt + 1) * TILE + row) * N + col))
        for e in range(TILE):
            acc += t.sload(base_a + buf + row * STRIDE + e) * t.sload(base_b + buf + e * STRIDE + col)
        yield
    t.gstore(C + row * N + col, acc)


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(11)
    A = rng.randn(M, K).astype(np.float64)
    B = rng.randn(K, N).astype(np.float64)
    ref = A @ B

    # Lower bound on shared-memory traffic: measured by actually running
    # the grader's own conflict-free kernel through the simulator.
    g_floor = GPU(M * N + M * K + K * N, smem_size=2 * 2 * TILE * STRIDE)
    a_base, b_base, c_base = M * N, M * N + M * K, 0
    g_floor.gmem[c_base:c_base + M * N] = 0.0
    g_floor.gmem[a_base:a_base + M * K] = A.ravel()
    g_floor.gmem[b_base:b_base + K * N] = B.ravel()
    m_floor = g_floor.launch(_conflict_free_reference, 1, TILE * TILE, a_base, b_base, c_base, K)
    floor = max(float(m_floor["smem_waves"]), 1.0)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "smem_wave_ratio": float("inf"), "error": str(e)}

    gpu = GPU(M * N + M * K + K * N, smem_size=2 * 2 * TILE * STRIDE)
    gpu.gmem[0:M * N] = 0.0
    gpu.gmem[M * N:M * N + M * K] = A.ravel()
    gpu.gmem[M * N + M * K:M * N + M * K + K * N] = B.ravel()

    params = {"C": 0, "A": M * N, "B": M * N + M * K, "M": M, "N": N, "K": K}
    try:
        m = prog.launch(gpu, "gemm_tile_dbuf", 1, TILE * TILE, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "smem_wave_ratio": float("inf"), "error": str(e)}

    out = gpu.gmem[0:M * N].reshape(M, N)
    max_err = float(np.max(np.abs(out - ref)))
    return {"max_abs_err": max_err, "smem_wave_ratio": float(m["smem_waves"]) / floor}


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
