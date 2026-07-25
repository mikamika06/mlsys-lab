"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Launches the SAME kernel twice against the same 16x16 matrices, once with
accumulate_fp16=0 (accumulator stays full precision) and once with
accumulate_fp16=1 (the accumulator itself is also rounded to fp16 mantissa
after every addition), and compares both against an exact float64 A @ B.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 16, 32


def _run(prog, A, B, accumulate_fp16):
    gpu = GPU(3 * N * N)
    gpu.gmem[0:N * N] = A.flatten()
    gpu.gmem[N * N:2 * N * N] = B.flatten()
    gpu.gmem[2 * N * N:3 * N * N] = 0.0
    params = {"A": 0, "B": N * N, "C": 2 * N * N, "N": N, "accumulate_fp16": accumulate_fp16}
    grid = (N * N + BLOCK - 1) // BLOCK
    prog.launch(gpu, "mixed_precision_matmul", grid, BLOCK, params)
    return gpu.gmem[2 * N * N:3 * N * N].reshape(N, N)


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(17)
    A = rng.uniform(-2.0, 2.0, size=(N, N))
    B = rng.uniform(-2.0, 2.0, size=(N, N))
    exact = A @ B
    denom = float(np.max(np.abs(exact)))

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()
    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"fp32_acc_err": float("inf"), "improvement": -1.0, "error": str(e)}

    try:
        C32 = _run(prog, A, B, 0)
        C16 = _run(prog, A, B, 1)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"fp32_acc_err": float("inf"), "improvement": -1.0, "error": str(e)}

    err32 = float(np.max(np.abs(C32 - exact))) / denom
    err16 = float(np.max(np.abs(C16 - exact))) / denom
    return {"fp32_acc_err": err32, "fp16_acc_err": err16, "improvement": err16 - err32}


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
