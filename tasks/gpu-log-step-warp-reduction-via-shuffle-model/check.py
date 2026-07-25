"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
each warp's reduced sum against a numpy oracle.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 64  # two warps, one block

OUT_BASE = 0
IN_BASE = 2
GMEM_SIZE = IN_BASE + N


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(4)
    x = rng.uniform(-2, 2, size=N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[OUT_BASE:OUT_BASE + 2] = -999.0
    gpu.gmem[IN_BASE:IN_BASE + N] = x

    params = {"out": OUT_BASE, "in": IN_BASE, "n": N}
    try:
        prog.launch(gpu, "warp_reduce_sum", 1, N, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref_out = np.array([x[0:32].sum(), x[32:64].sum()])
    max_err = float(np.max(np.abs(gpu.gmem[OUT_BASE:OUT_BASE + 2] - ref_out)))
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
