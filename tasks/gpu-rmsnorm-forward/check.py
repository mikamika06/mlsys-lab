"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the RMSNorm output against a numpy oracle.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = 32
EPS = 1e-5

OUT_BASE = 0
X_BASE = D
GAMMA_BASE = X_BASE + D
GMEM_SIZE = GAMMA_BASE + D


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(8)
    x = rng.uniform(-3, 3, size=D).astype(np.float64)
    gamma = rng.uniform(0.5, 2.0, size=D).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE, smem_size=D)
    gpu.gmem[OUT_BASE:OUT_BASE + D] = -999.0
    gpu.gmem[X_BASE:X_BASE + D] = x
    gpu.gmem[GAMMA_BASE:GAMMA_BASE + D] = gamma

    params = {"out": OUT_BASE, "x": X_BASE, "gamma": GAMMA_BASE, "eps": EPS, "n": D}
    try:
        prog.launch(gpu, "rmsnorm_forward", 1, D, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    rms = np.sqrt((x ** 2).mean() + EPS)
    ref_out = (x / rms) * gamma
    max_err = float(np.max(np.abs(gpu.gmem[OUT_BASE:OUT_BASE + D] - ref_out)))
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
