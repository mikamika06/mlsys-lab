"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the fused add+scale+ReLU output against a numpy oracle, and reports the
simulator's measured transaction count (a fused single-pass kernel
touches each of a[], b[], out[] exactly once per element; splitting the
op into separate add/scale/activation kernels round-trips an
intermediate buffer through global memory an extra time each).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 64, 32
SCALE = 1.5

OUT_BASE = 0
A_BASE = N
B_BASE = A_BASE + N
GMEM_SIZE = B_BASE + N


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(15)
    a = rng.uniform(-2, 2, size=N).astype(np.float64)
    b = rng.uniform(-2, 2, size=N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[OUT_BASE:OUT_BASE + N] = -999.0
    gpu.gmem[A_BASE:A_BASE + N] = a
    gpu.gmem[B_BASE:B_BASE + N] = b

    params = {"out": OUT_BASE, "a": A_BASE, "b": B_BASE, "scale": SCALE, "n": N}
    try:
        m = prog.launch(gpu, "fused_add_scale_relu", (N + BLOCK - 1) // BLOCK, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    ref_out = np.maximum(SCALE * (a + b), 0.0)
    max_err = float(np.max(np.abs(gpu.gmem[OUT_BASE:OUT_BASE + N] - ref_out)))
    return {"max_abs_err": max_err, "transactions": int(m["transactions"])}


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
