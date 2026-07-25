"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

A single 16x16x16 GEMM tile, one thread per output element (256 threads,
one block). The reference C is computed by an explicit Python triple loop
using the SAME left-to-right accumulation order the kernel uses, so a
numerically correct kernel matches it EXACTLY (max_abs_err == 0) regardless
of how its __shared__ tiles are laid out -- padding is a pure memory-layout
choice, it cannot change the arithmetic. What it changes is smem_waves: the
kernel's reduction reads As[row*lda+k] with `row` varying across every lane
of a warp, which conflicts badly on an unpadded (stride n) layout and is
conflict-free on a stride-(n+1) one.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 16  # tile / matrix dimension (single tile, no K-loop needed)


def _fixture():
    rng = np.random.RandomState(2026)
    A = rng.uniform(-1.0, 1.0, size=(N, N))
    B = rng.uniform(-1.0, 1.0, size=(N, N))
    return A, B


def _reference(A, B):
    # Same accumulation order (row-major, k left to right) a correct kernel
    # uses, so a correct kernel matches this EXACTLY, not just approximately.
    C = np.zeros((N, N))
    for row in range(N):
        for col in range(N):
            acc = 0.0
            for k in range(N):
                acc = acc + A[row, k] * B[k, col]
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
        return {"max_abs_err": float("inf"), "smem_waves": 10 ** 9, "error": str(e)}

    gpu = GPU(3 * N * N, smem_size=600)
    gpu.gmem[0:N * N] = A.flatten()            # A = gmem[0 : N*N]
    gpu.gmem[N * N:2 * N * N] = B.flatten()    # B = gmem[N*N : 2*N*N]
    gpu.gmem[2 * N * N:3 * N * N] = 0.0        # C = gmem[2*N*N : 3*N*N]

    params = {"C": 2 * N * N, "A": 0, "B": N * N, "n": N}
    try:
        m = prog.launch(gpu, "gemm_tile", 1, N * N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "smem_waves": 10 ** 9, "error": str(e)}

    C_got = gpu.gmem[2 * N * N:3 * N * N].reshape(N, N)
    max_err = float(np.max(np.abs(C_got - C_ref)))
    return {"max_abs_err": max_err, "smem_waves": int(m["smem_waves"])}


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
