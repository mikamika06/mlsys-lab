"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Checks the tree-summed total against a HIGH-PRECISION reference computed
with math.fsum (exact summation, immune to the very rounding error this
task is about) on a fixture with an extreme dynamic range -- one huge
value plus 1023 tiny ones -- where naive sequential accumulation loses
enough precision to fail the same tolerance.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 1024
TINY = 1e-15


def grade(srcfile: str = "solve.cu") -> dict:
    x = np.full(N, TINY, dtype=np.float64)
    x[0] = 1.0
    true_val = math.fsum(x.tolist())

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"rel_err": float("inf"), "error": str(e)}

    out_base = 0
    x_base = 1
    gpu = GPU(x_base + N, smem_size=N)
    gpu.gmem[out_base] = 0.0
    gpu.gmem[x_base:x_base + N] = x

    params = {"out": out_base, "x": x_base, "n": N}
    try:
        prog.launch(gpu, "tree_sum", 1, N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"rel_err": float("inf"), "error": str(e)}

    result = float(gpu.gmem[out_base])
    rel_err = abs(result - true_val) / abs(true_val)
    return {"rel_err": rel_err}


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
