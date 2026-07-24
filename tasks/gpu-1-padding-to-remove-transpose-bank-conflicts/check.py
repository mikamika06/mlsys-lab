"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Checks correctness against a numpy transpose oracle AND that the
shared-memory bank-conflict traffic matches a conflict-free lower bound the
grader computes itself (never hardcoded) by running its own padded-tile
kernel through the same simulator.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TILE = 32


def _conflict_free_reference(t, n):
    """Padded (TILE+1 stride) tile transpose, written directly against the
    simulator's native Thread API (not compiled from a .cu file) — this is
    only the grader's own measured lower bound on shared-memory traffic,
    independent of whatever solve.cu/ref.cu do."""
    base = t.salloc(TILE * (TILE + 1))
    row = t.threadIdx.x // TILE
    col = t.threadIdx.x % TILE
    t.sstore(base + row * (TILE + 1) + col, t.gload(row * n + col))
    yield
    t.gstore(row * n + col, t.sload(base + col * (TILE + 1) + row))


def grade(srcfile: str = "solve.cu") -> dict:
    n = TILE
    a = np.arange(n * n, dtype=np.float64)
    ref = a.reshape(n, n).T.ravel()

    # Lower bound on shared-memory traffic: measured by actually running
    # the grader's own conflict-free kernel through the simulator.
    g_floor = GPU(n * n, smem_size=TILE * (TILE + 1))
    g_floor.gmem[0:n * n] = a
    m_floor = g_floor.launch(_conflict_free_reference, 1, TILE * TILE, n)
    floor = max(float(m_floor["smem_waves"]), 1.0)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "smem_wave_ratio": float("inf"), "error": str(e)}

    gpu = GPU(2 * n * n, smem_size=TILE * (TILE + 1))
    gpu.gmem[0:n * n] = a
    gpu.gmem[n * n:2 * n * n] = 0.0

    params = {"out": n * n, "in": 0, "n": n}
    try:
        m = prog.launch(gpu, "transpose_tile", 1, TILE * TILE, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "smem_wave_ratio": float("inf"), "error": str(e)}

    out = gpu.gmem[n * n:2 * n * n]
    max_err = float(np.max(np.abs(out - ref)))
    return {"max_abs_err": max_err, "smem_wave_ratio": float(m["smem_waves"]) / floor}


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
