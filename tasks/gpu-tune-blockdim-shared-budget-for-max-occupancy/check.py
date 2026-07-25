"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Checks that the candidate's computed occupancy values reach the same
MAXIMUM achievable occupancy as a reference computed independently with
numpy across the same candidate blockDims.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK_DIMS = np.array([32, 64, 128, 256, 512, 1024], dtype=np.float64)
REGS_PER_THREAD = 32.0
SHARED_BYTES_PER_THREAD = 64.0
MAX_THREADS_PER_SM = 2048.0
MAX_BLOCKS_PER_SM = 32.0
MAX_REGS_PER_SM = 65536.0
MAX_SHARED_PER_SM = 98304.0


def grade(srcfile: str = "solve.cu") -> dict:
    n = len(BLOCK_DIMS)
    blocks_by_threads = np.floor(MAX_THREADS_PER_SM / BLOCK_DIMS)
    blocks_by_shared = np.floor(MAX_SHARED_PER_SM / (SHARED_BYTES_PER_THREAD * BLOCK_DIMS))
    blocks_by_regs = np.floor(MAX_REGS_PER_SM / (REGS_PER_THREAD * BLOCK_DIMS))
    actual_blocks = np.minimum(np.minimum(blocks_by_threads, blocks_by_shared),
                                np.minimum(blocks_by_regs, MAX_BLOCKS_PER_SM))
    ref_occ = (actual_blocks * BLOCK_DIMS) / MAX_THREADS_PER_SM
    ref_max = float(ref_occ.max())

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    out_base = 0
    bd_base = n
    gpu = GPU(bd_base + n)
    gpu.gmem[out_base:out_base + n] = 0.0
    gpu.gmem[bd_base:bd_base + n] = BLOCK_DIMS

    params = {"out": out_base, "block_dims": bd_base, "regs_per_thread": REGS_PER_THREAD,
              "shared_bytes_per_thread": SHARED_BYTES_PER_THREAD, "max_threads_per_sm": MAX_THREADS_PER_SM,
              "max_blocks_per_sm": MAX_BLOCKS_PER_SM, "max_regs_per_sm": MAX_REGS_PER_SM,
              "max_shared_per_sm": MAX_SHARED_PER_SM, "num_candidates": n}
    try:
        prog.launch(gpu, "compute_occupancy", 1, n, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    out = gpu.gmem[out_base:out_base + n]
    achieved_max = float(out.max())
    return {"max_abs_err": abs(achieved_max - ref_max)}


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
