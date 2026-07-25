"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the total modeled cycle cost against a numpy oracle.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 40, 32

OUT_BASE = 0
LEVEL_BASE = 1
LATENCY_BASE = LEVEL_BASE + N
GMEM_SIZE = LATENCY_BASE + 4


def grade(srcfile: str = "solve.cu") -> dict:
    # 10 register, 15 shared, 10 L2, 5 DRAM accesses, in that order.
    level = np.array([0] * 10 + [1] * 15 + [2] * 10 + [3] * 5, dtype=np.float64)
    latency = np.array([1.0, 25.0, 200.0, 450.0], dtype=np.float64)  # reg/shared/L2/DRAM

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[OUT_BASE] = -1.0
    gpu.gmem[LEVEL_BASE:LEVEL_BASE + N] = level
    gpu.gmem[LATENCY_BASE:LATENCY_BASE + 4] = latency

    params = {"out": OUT_BASE, "level": LEVEL_BASE, "latency": LATENCY_BASE, "n": N}
    try:
        prog.launch(gpu, "access_cost", 1, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref_total = float(latency[level.astype(np.int64)].sum())
    max_err = float(abs(gpu.gmem[OUT_BASE] - ref_total))
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
