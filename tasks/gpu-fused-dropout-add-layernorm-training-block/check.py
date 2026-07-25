"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). The
numpy oracle replicates the SAME deterministic arithmetic hash the kernel
must use for its "fixed-seed" dropout mask (there is no RNG builtin in this
language subset -- determinism comes from the formula itself, not a
stateful generator), then applies the same residual-add + LayerNorm.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R, D = 4, 16
N = R * D
DROPOUT_P, EPS, SEED = 0.3, 1e-5, 7


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(42)
    x = rng.randn(R, D).astype(np.float64)
    residual = rng.randn(R, D).astype(np.float64)
    gamma = (rng.randn(D).astype(np.float64) * 0.5 + 1.0)
    beta = rng.randn(D).astype(np.float64) * 0.1

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(3 * N + 2 * D, smem_size=256)
    off_out, off_x, off_res, off_gamma, off_beta = 0, N, 2 * N, 3 * N, 3 * N + D
    gpu.gmem[off_out:off_out + N] = -99.0
    gpu.gmem[off_x:off_x + N] = x.reshape(-1)
    gpu.gmem[off_res:off_res + N] = residual.reshape(-1)
    gpu.gmem[off_gamma:off_gamma + D] = gamma
    gpu.gmem[off_beta:off_beta + D] = beta

    params = {"out": off_out, "x": off_x, "residual": off_res, "gamma": off_gamma,
              "beta": off_beta, "d": D, "dropout_p": DROPOUT_P, "eps": EPS, "seed": SEED}
    try:
        prog.launch(gpu, "fused_block", R, D, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    got = gpu.gmem[off_out:off_out + N].reshape(R, D)

    idx_grid = np.arange(R)[:, None] * D + np.arange(D)[None, :]
    h = (idx_grid * 31 + SEED * 7 + 11) % 100
    keep = (h >= DROPOUT_P * 100.0).astype(np.float64)
    scale = 1.0 / (1.0 - DROPOUT_P)
    v = x * keep * scale + residual
    mean = v.mean(axis=1, keepdims=True)
    var = ((v - mean) ** 2).mean(axis=1, keepdims=True)
    ref = (v - mean) / np.sqrt(var + EPS) * gamma[None, :] + beta[None, :]

    max_err = float(np.max(np.abs(got - ref)))
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
