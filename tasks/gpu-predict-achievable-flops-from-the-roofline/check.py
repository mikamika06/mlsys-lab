"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the computed attainable FLOP/s against a numpy oracle (min(peak_flops,
ai * peak_bw)).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 5, 32

OUT_BASE = 0
FLOPS_BASE = N
BW_BASE = FLOPS_BASE + N
AI_BASE = BW_BASE + N
GMEM_SIZE = AI_BASE + N


def grade(srcfile: str = "solve.cu") -> dict:
    peak_flops = np.array([16e12, 16e12, 16e12, 312e12, 1e12], dtype=np.float64)
    peak_bw = np.array([2e12, 2e12, 2e12, 2e12, 500e9], dtype=np.float64)
    ai = np.array([2.0, 16.0, 8.0, 100.0, 0.5], dtype=np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[OUT_BASE:OUT_BASE + N] = -1.0
    gpu.gmem[FLOPS_BASE:FLOPS_BASE + N] = peak_flops
    gpu.gmem[BW_BASE:BW_BASE + N] = peak_bw
    gpu.gmem[AI_BASE:AI_BASE + N] = ai

    params = {"out": OUT_BASE, "peak_flops": FLOPS_BASE, "peak_bw": BW_BASE, "ai": AI_BASE, "n": N}
    try:
        prog.launch(gpu, "attainable_flops", 1, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref_out = np.minimum(peak_flops, ai * peak_bw)
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
