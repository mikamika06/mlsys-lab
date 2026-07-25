"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Checks
the derived per-step rescale factors against a numpy running-max oracle.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X = np.array([2.0, 1.0, 5.0, 3.0, 5.0, 6.0, 0.5, 6.0], dtype=np.float64)
N = len(X)


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(2 * N)
    gpu.gmem[0:N] = -1.0  # factors = gmem[0:N]
    gpu.gmem[N:2 * N] = X  # x = gmem[N:2N]

    params = {"factors": 0, "x": N, "n": N}
    try:
        prog.launch(gpu, "online_softmax_factors", 1, 1, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref_factors = np.ones(N)
    m = X[0]
    for i in range(1, N):
        new_m = max(m, X[i])
        ref_factors[i] = np.exp(m - new_m)
        m = new_m

    max_err = float(np.max(np.abs(gpu.gmem[0:N] - ref_factors)))
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
