"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Checks the warp-level inclusive scan against a per-warp np.cumsum oracle
over TWO warps (64 threads), so a scan that leaks across the 32-lane
warp boundary (e.g. computing `lane` from the wrong index) fails too.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 64  # two warps


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(9)
    x = rng.randn(N).astype(np.float64)
    ref = np.concatenate([np.cumsum(x[i:i + 32]) for i in range(0, N, 32)])

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(2 * N)
    gpu.gmem[0:N] = 0.0
    gpu.gmem[N:2 * N] = x

    params = {"out": 0, "in": N, "n": N}
    try:
        prog.launch(gpu, "warp_inclusive_scan", 1, N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    out = gpu.gmem[0:N]
    max_err = float(np.max(np.abs(out - ref)))
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
