"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the generated stream against a numpy oracle that runs the exact same
multiply-add-mod hash -- a counter-based RNG must reproduce the reference
stream bit-for-bit for the same (key, counter) pairs.
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
KEY = 12345.0

OUT_BASE = 0
COUNTERS_BASE = N
GMEM_SIZE = COUNTERS_BASE + N


def _oracle(counters, key):
    x = counters.astype(np.float64).copy()
    for r in range(3):
        x = np.fmod(x * 48271.0 + key + r * 7919.0, 1000003.0)
    return x / 1000003.0


def grade(srcfile: str = "solve.cu") -> dict:
    counters = np.arange(N, dtype=np.float64)  # thread i's own counter = i

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[OUT_BASE:OUT_BASE + N] = -1.0
    gpu.gmem[COUNTERS_BASE:COUNTERS_BASE + N] = counters

    params = {"out": OUT_BASE, "counters": COUNTERS_BASE, "key": KEY, "n": N}
    try:
        prog.launch(gpu, "philox_style_rng", (N + BLOCK - 1) // BLOCK, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref_out = _oracle(counters, KEY)
    max_err = float(np.max(np.abs(gpu.gmem[OUT_BASE:OUT_BASE + N] - ref_out)))
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
