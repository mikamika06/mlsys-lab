"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

n = 100 is not a multiple of blockDim.x * gridDim.x = 32 * 4 = 128, so the
last block's threads 100..127 must not touch memory at all. `out` is
pre-filled with a sentinel before launch; the oracle expects that sentinel
to survive untouched past index n. A kernel whose tail mask doesn't
actually exclude those threads overwrites the sentinel with `s * in[i]` for
i in [n, 128) instead, which the grader catches by comparing the WHOLE
128-element output buffer, not just the first 100 elements.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK, GRID = 100, 32, 4  # BLOCK * GRID = 128 > N -- last block has a tail
TOTAL = BLOCK * GRID
S = 3.0
SENTINEL = -999.0


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(11)
    x = rng.uniform(-5.0, 5.0, size=TOTAL)

    ref = np.full(TOTAL, SENTINEL)
    ref[:N] = S * x[:N]

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(2 * TOTAL)
    gpu.gmem[0:TOTAL] = x               # in  = gmem[0:TOTAL]
    gpu.gmem[TOTAL:2 * TOTAL] = SENTINEL  # out = gmem[TOTAL:2*TOTAL], pre-filled

    params = {"out": TOTAL, "in": 0, "n": N, "s": S}
    try:
        prog.launch(gpu, "scale_masked", GRID, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    got = gpu.gmem[TOTAL:2 * TOTAL]
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
