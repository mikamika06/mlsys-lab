"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

`bias` is allocated with EXACTLY c elements -- never materialized to the
full r*c size -- with a "trap" region of recognizable sentinel values
placed immediately after it in gmem. A kernel that indexes `bias` as if it
had already been expanded to n elements (no `% c`) reads straight into the
trap for every row past the first, which the grader catches by comparing
the full output.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R, C = 20, 8
N = R * C
BLOCK, GRID = 32, 5  # 160 threads == N, no tail to worry about here
TRAP = 777.0


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(51)
    a = rng.uniform(-3.0, 3.0, size=(R, C))
    bias = rng.uniform(-1.0, 1.0, size=C)
    ref = (a + bias[None, :]).flatten()

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    trap_len = N - C
    gpu = GPU(N + N + C + trap_len)
    gpu.gmem[0:N] = -1.0                          # out  = gmem[0:N]
    gpu.gmem[N:2 * N] = a.flatten()                # a    = gmem[N:2N]
    gpu.gmem[2 * N:2 * N + C] = bias                # bias = gmem[2N:2N+C], exactly C elements
    gpu.gmem[2 * N + C:2 * N + C + trap_len] = TRAP  # trap region right after bias

    params = {"out": 0, "a": N, "bias": 2 * N, "r": R, "c": C, "n": N}
    try:
        prog.launch(gpu, "broadcast_add", GRID, BLOCK, params)
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
