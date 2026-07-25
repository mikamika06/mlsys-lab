"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). The
oracle recomputes the same hash-based mask independently in Python.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK, GRID = 200, 32, 7  # 224 threads > N, exercises the tail too
SEED = 12345
P = 0.3


def _reference():
    mask = np.empty(N)
    for i in range(N):
        h = (SEED + i * 2654435761) % 1000000007
        u = h / 1000000007.0
        mask[i] = 0.0 if u < P else 1.0
    return mask


def grade(srcfile: str = "solve.cu") -> dict:
    ref = _reference()

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(N)
    gpu.gmem[0:N] = -1.0  # mask, pre-filled -- every in-range element must be overwritten

    params = {"mask": 0, "n": N, "seed": SEED, "p": P}
    try:
        prog.launch(gpu, "dropout_mask", GRID, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    got = gpu.gmem[0:N]
    max_err = float(np.max(np.abs(got - ref)))
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
