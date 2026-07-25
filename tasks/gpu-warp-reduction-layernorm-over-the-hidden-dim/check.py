"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU), using
the simulator's real warp-synchronous __shfl_xor_sync implementation.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS, D, BLOCK = 4, 128, 32  # one warp (32 threads) per row, D/32 = 4 elements/thread
EPS = 1e-5


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(41)
    x = rng.uniform(-3.0, 3.0, size=(ROWS, D))
    gamma = rng.uniform(0.5, 1.5, size=D)
    beta = rng.uniform(-0.5, 0.5, size=D)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()
    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(ROWS * D * 2 + D * 2)
    gpu.gmem[0:ROWS * D] = x.flatten()
    gpu.gmem[ROWS * D:ROWS * D + D] = gamma
    gpu.gmem[ROWS * D + D:ROWS * D + 2 * D] = beta
    gpu.gmem[ROWS * D + 2 * D:ROWS * D + 2 * D + ROWS * D] = 0.0
    params = {"x": 0, "gamma": ROWS * D, "beta": ROWS * D + D,
              "y": ROWS * D + 2 * D, "rows": ROWS, "D": D, "eps": EPS}

    try:
        prog.launch(gpu, "warp_layernorm", ROWS, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    y = gpu.gmem[ROWS * D + 2 * D:ROWS * D + 2 * D + ROWS * D].reshape(ROWS, D)
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    expected = (x - mean) / np.sqrt(var + EPS) * gamma + beta
    max_err = float(np.max(np.abs(y - expected)))
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
