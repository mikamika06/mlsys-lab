"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

`in` has exactly n real elements; immediately after it in gmem sits a
"trap" region filled with a large, recognizable value. A kernel that
doesn't mask its load of `in` reads straight into the trap for the tail
threads, which the grader catches by comparing the whole padded `out`
buffer against a reference that expects 0.0 there.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK, GRID = 90, 32, 3  # BLOCK * GRID = 96 > N -- last block has a 6-thread tail
TOTAL = BLOCK * GRID
S = 4.0
TRAP = 777.0


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(31)
    x = rng.uniform(-5.0, 5.0, size=N)

    ref = np.zeros(TOTAL)
    ref[:N] = S * x

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    trap_len = TOTAL - N
    gpu = GPU(TOTAL + N + trap_len)
    gpu.gmem[0:TOTAL] = -1.0            # out = gmem[0:TOTAL], pre-filled (must be fully overwritten)
    gpu.gmem[TOTAL:TOTAL + N] = x       # in  = gmem[TOTAL:TOTAL+N]  (exactly N real elements)
    gpu.gmem[TOTAL + N:TOTAL + N + trap_len] = TRAP  # trap region right after `in`

    params = {"out": 0, "in": TOTAL, "n": N, "s": S}
    try:
        prog.launch(gpu, "masked_scale_fill", GRID, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    got = gpu.gmem[0:TOTAL]
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
