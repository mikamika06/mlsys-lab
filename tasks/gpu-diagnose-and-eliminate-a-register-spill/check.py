"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Real
register spills lower into loads/stores against per-thread "local memory",
which is physically backed by global memory -- so this grader models a spill
exactly as extra global-memory traffic through a `scratch` buffer, and reads
the real transaction count straight from the simulator's coalescing model.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 128, 32


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(9)
    x = rng.uniform(1.0, 5.0, size=N)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    gpu = GPU(N + N * 4 + N)
    gpu.gmem[0:N] = x
    gpu.gmem[N:N + N * 4] = 0.0
    gpu.gmem[N + N * 4:N + N * 4 + N] = 0.0
    params = {"x": 0, "scratch": N, "out": N + N * 4, "n": N}

    try:
        m = prog.launch(gpu, "compute_expr", (N + BLOCK - 1) // BLOCK, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    out = gpu.gmem[N + N * 4:N + N * 4 + N]
    expected = ((x + 1.0) * (x + 2.0) - (x + 3.0)) / (x + 4.0)
    max_err = float(np.max(np.abs(out - expected)))
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
