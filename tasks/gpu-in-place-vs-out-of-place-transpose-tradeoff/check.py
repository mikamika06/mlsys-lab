"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute BOTH of its kernels thread-by-thread on the software GPU
(arena.cuda_sim.GPU) -- this CUDA-C subset allows several distinctly-named
__global__ functions in one source file, just not two sharing a name.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 16


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(4)
    A = rng.uniform(-1.0, 1.0, size=(N, N))
    ref = A.T

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"in_place_max_abs_err": float("inf"), "out_of_place_max_abs_err": float("inf"),
                "error": str(e)}

    result = {}

    # -- in-place: single buffer, no second array.
    try:
        gpu = GPU(N * N)
        gpu.gmem[0:N * N] = A.flatten()
        m1 = prog.launch(gpu, "transpose_in_place", 1, N * N, {"A": 0, "n": N})
        got = gpu.gmem[0:N * N].reshape(N, N)
        result["in_place_max_abs_err"] = float(np.max(np.abs(got - ref)))
        result["in_place_transactions"] = int(m1["transactions"])
    except Exception as e:  # noqa: BLE001
        result["in_place_max_abs_err"] = float("inf")
        result["in_place_error"] = str(e)

    # -- out-of-place: separate output buffer, input left untouched.
    try:
        gpu2 = GPU(2 * N * N)
        gpu2.gmem[0:N * N] = A.flatten()
        gpu2.gmem[N * N:2 * N * N] = 0.0
        m2 = prog.launch(gpu2, "transpose_out_of_place", 1, N * N,
                          {"out": N * N, "in": 0, "n": N})
        got2 = gpu2.gmem[N * N:2 * N * N].reshape(N, N)
        result["out_of_place_max_abs_err"] = float(np.max(np.abs(got2 - ref)))
        result["out_of_place_transactions"] = int(m2["transactions"])
    except Exception as e:  # noqa: BLE001
        result["out_of_place_max_abs_err"] = float("inf")
        result["out_of_place_error"] = str(e)

    return result


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
