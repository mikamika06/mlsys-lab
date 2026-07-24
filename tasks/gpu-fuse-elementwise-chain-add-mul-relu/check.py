"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Compares the resulting gmem against a numpy reference and reports the
transaction count the simulator observed (a fused, one-pass-per-element
kernel touches global memory far less than one that round-trips an
intermediate through `out`).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 256, 64


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(7)
    a = rng.randn(N).astype(np.float64)
    b = rng.randn(N).astype(np.float64)
    c = rng.randn(N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    gpu = GPU(4 * N)
    gpu.gmem[0 * N:1 * N] = 0.0   # out = gmem[0:N]      (poisoned zero, not the answer)
    gpu.gmem[1 * N:2 * N] = a     # a   = gmem[N:2N]
    gpu.gmem[2 * N:3 * N] = b     # b   = gmem[2N:3N]
    gpu.gmem[3 * N:4 * N] = c     # c   = gmem[3N:4N]

    params = {"out": 0, "a": N, "b": 2 * N, "c": 3 * N, "n": N}
    try:
        m = prog.launch(gpu, "fuse", N // BLOCK, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    ref_out = np.maximum((a + b) * c, 0.0)
    max_err = float(np.max(np.abs(gpu.gmem[0:N] - ref_out)))
    return {"max_abs_err": max_err, "transactions": int(m["transactions"])}


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
