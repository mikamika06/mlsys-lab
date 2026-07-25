"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
a 32-lane (one warp) divergent-branch-then-shuffle kernel against a numpy
oracle -- a warp shuffle placed right after a divergent `if`, with no explicit
synchronization before it, can read a lane that hasn't reached the shuffle
yet (this simulator has no real hardware, but its round-robin scheduler
reproduces the SAME class of bug deterministically: threads that took a path
with an extra synchronization point arrive at the shuffle a full round later
than threads that didn't, so the shuffle's "neighbor" value is whatever that
lane published in ITS round -- which for the boundary lane is nothing yet).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 32  # exactly one warp


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(9)
    x = rng.randn(N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(2 * N, smem_size=1)
    gpu.gmem[0:N] = x            # in  = gmem[0:N]
    gpu.gmem[N:2 * N] = 0.0      # out = gmem[N:2N]

    params = {"out": N, "in": 0, "n": N}
    try:
        prog.launch(gpu, "divergent_shuffle", 1, N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    val = x.copy()
    val[:16] *= 2.0             # the divergent branch: lanes 0..15 double their value
    ref_out = np.empty(N)
    ref_out[0] = val[0]         # shfl_up with no lane below: keeps its own value
    ref_out[1:] = val[:-1]      # every other lane reads its neighbor one below

    max_err = float(np.max(np.abs(gpu.gmem[N:2 * N] - ref_out)))
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
