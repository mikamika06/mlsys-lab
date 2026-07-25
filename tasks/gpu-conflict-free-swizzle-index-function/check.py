"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). The
simulator's own shared-memory bank-conflict model (arena.cuda_sim.GPU,
BANKS=32) is the sole source of `smem_waves` -- not a hand-rolled invariant
check -- so a genuinely conflict-free physical layout is the only way to pass.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 32          # tile is N x N, one warp (N threads, N == WARP)
TARGET_COL = 7


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(5)
    A = rng.uniform(-1.0, 1.0, size=(N, N))

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "smem_waves": 10 ** 9, "error": str(e)}

    gpu = GPU(N * N + N, smem_size=N * N)
    gpu.gmem[0:N * N] = A.flatten()
    gpu.gmem[N * N:N * N + N] = 0.0
    params = {"in": 0, "out": N * N, "target_col": TARGET_COL}

    try:
        m = prog.launch(gpu, "swizzle_roundtrip", 1, N, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "smem_waves": 10 ** 9, "error": str(e)}

    out = gpu.gmem[N * N:N * N + N]
    expected = A[:, TARGET_COL]
    max_err = float(np.max(np.abs(out - expected)))
    return {"max_abs_err": max_err, "smem_waves": int(m["smem_waves"])}


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
