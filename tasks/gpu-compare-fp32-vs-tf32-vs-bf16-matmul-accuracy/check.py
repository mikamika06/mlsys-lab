"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

Launches the SAME kernel three times against the same 8x8 matrices, once per
mantissa width (fp32 = 23 explicit bits, tf32 = 10, bf16 = 7), and compares
each run's output against an exact numpy (float64) matmul -- the reference
this simulator's arithmetic is already computed in, so no separate "fp64
kernel" is needed.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 8
BLOCK = 32
FP32_BITS, TF32_BITS, BF16_BITS = 23.0, 10.0, 7.0


def _build_matrices():
    rng = np.random.RandomState(11)
    A = rng.uniform(-2.0, 2.0, size=(N, N))
    B = rng.uniform(-2.0, 2.0, size=(N, N))
    # Keep every element comfortably away from zero (logf(0) is undefined,
    # and the point of this fixture is dynamic-range-agnostic rounding).
    A[np.abs(A) < 0.2] = 0.5
    B[np.abs(B) < 0.2] = 0.5
    return A, B


def _run(prog, A, B, mantissa_bits):
    gpu = GPU(3 * N * N)
    gpu.gmem[0:N * N] = A.flatten()
    gpu.gmem[N * N:2 * N * N] = B.flatten()
    gpu.gmem[2 * N * N:3 * N * N] = 0.0
    params = {"A": 0, "B": N * N, "C": 2 * N * N, "N": N, "mantissa_bits": mantissa_bits}
    grid = (N * N + BLOCK - 1) // BLOCK
    prog.launch(gpu, "quantized_matmul", grid, BLOCK, params)
    return gpu.gmem[2 * N * N:3 * N * N].reshape(N, N)


def grade(srcfile: str = "solve.cu") -> dict:
    A, B = _build_matrices()
    exact = A @ B  # float64 -- this simulator's own arithmetic is already fp64
    denom = float(np.max(np.abs(exact)))

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()
    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"fp32_rel_err": float("inf"), "tf32_rel_err": float("inf"),
                 "bf16_rel_err": float("inf"), "error": str(e)}

    out = {}
    for name, bits in (("fp32", FP32_BITS), ("tf32", TF32_BITS), ("bf16", BF16_BITS)):
        try:
            C = _run(prog, A, B, bits)
            rel_err = float(np.max(np.abs(C - exact))) / denom
        except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
            rel_err = float("inf")
        out[f"{name}_rel_err"] = rel_err
    return out


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
