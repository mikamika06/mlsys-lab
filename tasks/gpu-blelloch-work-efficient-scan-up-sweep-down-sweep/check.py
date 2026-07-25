"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU). Runs a single 32-thread
block that computes an exclusive prefix sum via up-sweep/down-sweep in shared
memory, and compares gmem against a numpy oracle (np.cumsum shifted by one).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 32


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(7)
    x = rng.uniform(-5.0, 5.0, size=N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(2 * N, smem_size=N)
    gpu.gmem[0:N] = x        # in  = gmem[0:N]
    gpu.gmem[N:2 * N] = 0.0  # out = gmem[N:2N]

    params = {"out": N, "in": 0, "n": N}
    try:
        prog.launch(gpu, "scan", 1, N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref_out = np.zeros(N, dtype=np.float64)
    ref_out[1:] = np.cumsum(x)[:-1]  # exclusive scan: out[i] = sum(x[0..i))
    max_err = float(np.max(np.abs(gpu.gmem[N:2 * N] - ref_out)))
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
