"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Checks the segmented scan against a reference computed with numpy: split
the 32 elements into segments by head_flag, cumsum independently within
each segment.
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
    rng = np.random.RandomState(11)
    x = rng.randn(N).astype(np.float64)
    flags = np.zeros(N)
    for h in (0, 5, 6, 13, 20, 21, 22, 30):
        flags[h] = 1.0

    seg_id = np.cumsum(flags) - 1
    ref = np.zeros_like(x)
    for s in range(int(seg_id.max()) + 1):
        idx = seg_id == s
        ref[idx] = np.cumsum(x[idx])

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    out_base = 0
    in_base = N
    flag_base = 2 * N
    gpu = GPU(3 * N)
    gpu.gmem[out_base:out_base + N] = 0.0
    gpu.gmem[in_base:in_base + N] = x
    gpu.gmem[flag_base:flag_base + N] = flags

    params = {"out": out_base, "in": in_base, "head_flag": flag_base, "n": N}
    try:
        prog.launch(gpu, "segmented_scan", 1, N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    out = gpu.gmem[out_base:out_base + N]
    max_err = float(np.max(np.abs(out - ref)))
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
