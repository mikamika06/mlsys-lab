"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU). Runs a batched online
softmax (one thread per row, running-max/running-sum single pass) and
compares against a numpy batch-softmax oracle (scipy-free, computed directly
from the same seeded input).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
B, N = 8, 32


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(21)
    x = rng.uniform(-6.0, 6.0, size=(B, N))

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(2 * B * N)
    gpu.gmem[0:B * N] = x.reshape(-1)         # in  = gmem[0 : B*N]
    gpu.gmem[B * N:2 * B * N] = 0.0            # out = gmem[B*N : 2*B*N]

    params = {"out": B * N, "in": 0, "B": B, "N": N}
    try:
        prog.launch(gpu, "online_softmax", 1, B, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ex = np.exp(x - x.max(axis=1, keepdims=True))
    oracle = ex / ex.sum(axis=1, keepdims=True)
    got = gpu.gmem[B * N:2 * B * N].reshape(B, N)
    max_err = float(np.max(np.abs(got - oracle)))
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
