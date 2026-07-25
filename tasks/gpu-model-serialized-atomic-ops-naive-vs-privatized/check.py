"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

Real atomicAdd has no builtin in this CUDA-C subset (nor does any real
concurrency-driven race actually occur in this simulator's execution
model), so "atomic operation count" here is a MODELED, per-configuration
formula the kernel itself computes: how many global atomic ops each of two
histogram-update strategies needs for n[i] updates over block_size[i]-
thread blocks. check.py computes the same formula independently (with
Python's own ceiling division, not by reading the kernel's arithmetic) as
the oracle.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [
    (100000.0, 256.0),
    (100000.0, 32.0),
    (1000.0, 128.0),
    (7.0, 32.0),      # fewer updates than one block's threads
    (99328.0, 256.0),  # n an exact multiple of block_size (388 * 256)
]
M = len(CONFIGS)


def grade(srcfile: str = "solve.cu") -> dict:
    n_vals = np.array([c[0] for c in CONFIGS])
    bs_vals = np.array([c[1] for c in CONFIGS])
    naive_ref = n_vals.copy()
    privatized_ref = np.ceil(n_vals / bs_vals)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(4 * M)
    gpu.gmem[0 * M:1 * M] = 0.0
    gpu.gmem[1 * M:2 * M] = 0.0
    gpu.gmem[2 * M:3 * M] = n_vals
    gpu.gmem[3 * M:4 * M] = bs_vals

    params = {"naive_out": 0 * M, "privatized_out": 1 * M, "n": 2 * M, "block_size": 3 * M, "m": M}
    try:
        prog.launch(gpu, "modeled_atomic_counts", 1, M, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    naive_got = gpu.gmem[0 * M:1 * M]
    privatized_got = gpu.gmem[1 * M:2 * M]
    err = max(float(np.max(np.abs(naive_got - naive_ref))),
              float(np.max(np.abs(privatized_got - privatized_ref))))
    return {"max_abs_err": err}


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
