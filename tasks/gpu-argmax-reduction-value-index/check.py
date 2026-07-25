"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the reduced (max value, argmax index) pair against a numpy oracle, on a fixed
fixture that deliberately contains a tie to pin down the lowest-index rule.
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


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(123)
    x = rng.randn(N).astype(np.float64)
    # Force a tie between two distinct indices to pin down the tie rule.
    x[3] = 5.0
    x[27] = 5.0

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "index_exact": 0.0, "error": str(e)}

    gpu = GPU(N + 2, smem_size=64)
    gpu.gmem[0:N] = x
    gpu.gmem[N:N + 2] = -1.0

    params = {"in": 0, "out": N, "n": N}
    try:
        prog.launch(gpu, "argmax_reduce", 1, N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "index_exact": 0.0, "error": str(e)}

    ref_val = float(np.max(x))
    ref_idx = float(np.argmax(x))  # numpy argmax already returns the FIRST (lowest) index on ties

    got_val = float(gpu.gmem[N])
    got_idx = float(gpu.gmem[N + 1])

    return {
        "max_abs_err": abs(got_val - ref_val),
        "index_exact": 1.0 if got_idx == ref_idx else 0.0,
    }


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
