"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

No float4 type exists in this CUDA-C subset, so "load instruction count"
is modeled explicitly: the kernel itself reports, per thread, whether it
did a 4-wide group load (1) or nothing (0) via `load_flag`. check.py sums
that report and compares it to the closed-form floor(n/4) + (n%4) --
independently computed, not read back out of the kernel's own reporting.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 50, 32  # floor(50/4) + 50%4 = 12 + 2 = 14 <= 32 threads


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(41)
    x = rng.uniform(-3.0, 3.0, size=N)
    expected_loads = N // 4 + N % 4

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "load_count_err": float("inf"), "error": str(e)}

    gpu = GPU(N + BLOCK + N)
    gpu.gmem[0:N] = -1.0                    # out       = gmem[0:N]
    gpu.gmem[N:N + BLOCK] = -1.0             # load_flag = gmem[N:N+BLOCK]
    gpu.gmem[N + BLOCK:N + BLOCK + N] = x    # in        = gmem[N+BLOCK:N+BLOCK+N]

    params = {"out": 0, "load_flag": N, "in": N + BLOCK, "n": N}
    try:
        prog.launch(gpu, "vectorized_copy_load_count", 1, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "load_count_err": float("inf"), "error": str(e)}

    out_got = gpu.gmem[0:N]
    load_flag_got = gpu.gmem[N:N + BLOCK]
    max_err = float(np.max(np.abs(out_got - x)))
    load_count_err = abs(float(np.sum(load_flag_got)) - expected_loads)
    return {"max_abs_err": max_err, "load_count_err": load_count_err}


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
