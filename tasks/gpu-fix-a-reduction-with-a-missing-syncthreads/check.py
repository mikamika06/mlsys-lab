"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
an 8-element shared-memory sum reduction against a numpy oracle -- a missing
__syncthreads() between reduction steps lets a thread race ahead into the next
step and read a shared-memory slot another thread hasn't finished updating
yet, corrupting the sum deterministically (this simulator has no real
concurrency, so it's not a flaky race -- the wrong answer is exactly
reproducible, which is what makes it gradeable).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 8


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(5)
    x = rng.randn(N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(N + 1, smem_size=8)
    gpu.gmem[0:N] = x   # in  = gmem[0:N]
    gpu.gmem[N] = 0.0    # out = gmem[N]

    params = {"out": N, "in": 0, "n": N}
    try:
        prog.launch(gpu, "sum_reduce", 1, N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref_out = float(x.sum())
    max_err = float(abs(gpu.gmem[N] - ref_out))
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
