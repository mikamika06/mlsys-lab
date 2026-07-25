"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU). Compares the modeled
mixed-precision dot product against a full-fp32 numpy reference, within a
tolerance that accounts for the expected quantization error (not exact --
this is precision loss, not a bug).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 64


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(9)
    a = rng.uniform(-2.0, 2.0, N)
    b = rng.uniform(-2.0, 2.0, N)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(1 + 2 * N)
    gpu.gmem[0] = -999.0
    gpu.gmem[1:1 + N] = a
    gpu.gmem[1 + N:1 + 2 * N] = b

    try:
        prog.launch(gpu, "half2_matmul_dot", 1, 32, {"out": 0, "a": 1, "b": 1 + N, "n": N})
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref = float(np.dot(a, b))
    max_err = float(abs(gpu.gmem[0] - ref))
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
