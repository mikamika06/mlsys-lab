"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
both the forward-dropout output and the backward-recomputed gradient
against a numpy oracle -- the backward pass must reproduce the forward
mask exactly, with no mask buffer passed between them.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 64, 32  # 2 blocks -- exercises the global index, not just threadIdx.x
SEED = 777.0
KEEP_PROB = 0.7

FWD_BASE = 0
BWD_BASE = N
X_BASE = BWD_BASE + N
GRAD_BASE = X_BASE + N
GMEM_SIZE = GRAD_BASE + N


def _hash01(i, seed):
    h = i.astype(np.float64).copy()
    for r in range(3):
        h = np.fmod(h * 48271.0 + seed + r * 7919.0, 1000003.0)
    return h / 1000003.0


def grade(srcfile: str = "solve.cu") -> dict:
    i = np.arange(N)
    x = i.astype(np.float64) * 0.1 + 1.0
    grad_in = i.astype(np.float64) * 0.05 + 2.0

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[FWD_BASE:FWD_BASE + N] = -1.0
    gpu.gmem[BWD_BASE:BWD_BASE + N] = -1.0
    gpu.gmem[X_BASE:X_BASE + N] = x
    gpu.gmem[GRAD_BASE:GRAD_BASE + N] = grad_in

    params = {"fwd_out": FWD_BASE, "bwd_grad": BWD_BASE, "x": X_BASE,
              "grad_in": GRAD_BASE, "seed": SEED, "keep_prob": KEEP_PROB, "n": N}
    try:
        prog.launch(gpu, "dropout_fwd_bwd", (N + BLOCK - 1) // BLOCK, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    rand01 = _hash01(i, SEED)
    keep = (rand01 < KEEP_PROB).astype(np.float64)
    ref_fwd = keep * x / KEEP_PROB
    ref_bwd = keep * grad_in / KEEP_PROB

    fwd_err = float(np.max(np.abs(gpu.gmem[FWD_BASE:FWD_BASE + N] - ref_fwd)))
    bwd_err = float(np.max(np.abs(gpu.gmem[BWD_BASE:BWD_BASE + N] - ref_bwd)))
    return {"max_abs_err": max(fwd_err, bwd_err)}


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
