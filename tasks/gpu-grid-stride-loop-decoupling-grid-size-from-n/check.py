"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU) at
THREE different grid sizes for the same n -- one that launches at least n
threads, and two that launch far fewer, requiring each thread to loop over
several elements. A kernel that only handles one element per thread passes
the first configuration (by luck: it happens to launch enough threads) and
fails the other two, which a single fixed grid size could never catch.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 200
BLOCK = 32
GRID_CONFIGS = [7, 2, 1]  # 224, 64, 32 threads -- only the first covers N in one pass
S = 2.0


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(9)
    x = rng.uniform(-3.0, 3.0, size=N)
    ref = S * x

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    worst = 0.0
    for grid in GRID_CONFIGS:
        gpu = GPU(2 * N)
        gpu.gmem[0:N] = x            # in  = gmem[0:N]
        gpu.gmem[N:2 * N] = -999.0   # out = gmem[N:2N], pre-filled sentinel
        params = {"out": N, "in": 0, "n": N, "s": S}
        try:
            prog.launch(gpu, "scale_grid_stride", grid, BLOCK, params)
        except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
            return {"max_abs_err": float("inf"), "error": f"grid={grid}: {e}"}
        got = gpu.gmem[N:2 * N]
        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}


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
