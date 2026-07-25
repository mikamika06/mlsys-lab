"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the computed variance against a numpy oracle (np.var, computed with a
numerically stable two-pass algorithm) on a large-mean fixture where the
naive E[x^2]-E[x]^2 formula catastrophically cancels.
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
MEAN = 1e10


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(5)
    eps = rng.randint(-3, 4, size=N).astype(np.float64)
    x = MEAN + eps

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(N + 1, smem_size=64)
    gpu.gmem[0] = 0.0
    gpu.gmem[1:1 + N] = x

    params = {"out": 0, "x": 1, "n": N}
    try:
        prog.launch(gpu, "row_variance", 1, N, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref_var = float(np.var(x))  # numpy's variance is computed via a stable two-pass algorithm
    max_err = float(abs(gpu.gmem[0] - ref_var))
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
