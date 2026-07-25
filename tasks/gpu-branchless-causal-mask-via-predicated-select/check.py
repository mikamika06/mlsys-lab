"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU). Applies a causal mask to
an n x n score matrix and checks both the output VALUES (against a numpy
oracle) and the simulator's `divergences` count (warps whose lanes issued a
different number of global-memory accesses -- the access-count proxy for
control-flow divergence this simulator measures).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 64
NEG_INF = -1.0e30


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(11)
    score = rng.randn(N * N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "divergences": 10 ** 9, "error": str(e)}

    gpu = GPU(2 * N * N)
    gpu.gmem[0:N * N] = score        # score = gmem[0 : N*N]
    gpu.gmem[N * N:2 * N * N] = 0.0  # out   = gmem[N*N : 2*N*N]

    params = {"out": N * N, "score": 0, "n": N}
    try:
        m = prog.launch(gpu, "causal_mask", N, N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "divergences": 10 ** 9, "error": str(e)}

    idx = np.arange(N * N)
    i_idx, j_idx = idx // N, idx % N
    oracle = np.where(j_idx <= i_idx, score, NEG_INF)
    max_err = float(np.max(np.abs(gpu.gmem[N * N:2 * N * N] - oracle)))
    return {"max_abs_err": max_err, "divergences": int(m["divergences"])}


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
