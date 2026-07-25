"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Checks that the classification label written to `out` matches a reference
computed directly from the same priority rule, applied with numpy (never
hardcoded per-case).
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
    # All 4 combinations of the two flags, doubled to fill N=8 -- covers
    # the priority-order case (both flags set) as well as each flag alone.
    cross_block = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.float64)
    shared_target_unknown = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(3 * N)
    gpu.gmem[0:N] = 0.0
    gpu.gmem[N:2 * N] = cross_block
    gpu.gmem[2 * N:3 * N] = shared_target_unknown

    params = {"out": 0, "cross_block": N, "shared_target_unknown": 2 * N, "n": N}
    try:
        m = prog.launch(gpu, "classify_sync_strategy", 1, N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    # Reference COMPUTED from the same rule, not a hardcoded label list:
    # data-dependent collision (shared_target_unknown) takes priority --
    # a global atomicAdd already reaches every block, so it subsumes the
    # need for a separate two-pass combine.
    ref = np.where(shared_target_unknown > 0.5, 1.0,
                    np.where(cross_block > 0.5, 2.0, 0.0))

    out = gpu.gmem[0:N]
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
