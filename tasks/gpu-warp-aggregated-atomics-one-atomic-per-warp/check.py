"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

Real __ballot_sync-based leader election has no builtin in this CUDA-C
subset, so warp aggregation is modeled the honest way available: each
configuration IS one warp, described by how many of its 32 lanes want to
increment (`active_count[i]`), and the kernel reports the two strategies'
atomic-op counts for it directly.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# One entry per WARP: how many of its 32 lanes want to increment.
ACTIVE_COUNTS = [32.0, 32.0, 1.0, 1.0, 1.0, 0.0, 16.0, 8.0, 4.0, 20.0]
M = len(ACTIVE_COUNTS)


def grade(srcfile: str = "solve.cu") -> dict:
    active = np.array(ACTIVE_COUNTS)
    naive_ref = active.copy()
    warp_agg_ref = (active > 0.5).astype(float)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(3 * M)
    gpu.gmem[0 * M:1 * M] = 0.0
    gpu.gmem[1 * M:2 * M] = 0.0
    gpu.gmem[2 * M:3 * M] = active

    params = {"naive_out": 0 * M, "warp_agg_out": 1 * M, "active_count": 2 * M, "m": M}
    try:
        prog.launch(gpu, "warp_aggregated_atomic_counts", 1, M, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    naive_got = gpu.gmem[0 * M:1 * M]
    warp_agg_got = gpu.gmem[1 * M:2 * M]
    err = max(float(np.max(np.abs(naive_got - naive_ref))),
              float(np.max(np.abs(warp_agg_got - warp_agg_ref))))
    return {"max_abs_err": err}


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
