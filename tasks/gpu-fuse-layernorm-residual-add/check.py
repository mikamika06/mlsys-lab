"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Checks LayerNorm(x + residual) * gamma + beta correctness against numpy,
AND that global-memory traffic is genuinely lower than an UNFUSED
two-kernel baseline the grader runs itself (via the simulator's native
Thread API, not compiled from a .cu file) -- the unfused baseline writes
x+residual to a separate global array and reads it back, exactly what a
truly fused kernel must avoid.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, D = 32, 8
EPS = 1e-5


def _unfused_add_kernel(t, x_base, r_base, tmp_base, N_, D_):
    i = t.threadIdx.x
    if i < N_:
        for d in range(D_):
            v = t.gload(x_base + i * D_ + d) + t.gload(r_base + i * D_ + d)
            t.gstore(tmp_base + i * D_ + d, v)


def _unfused_layernorm_kernel(t, out_base, tmp_base, g_base, b_base, N_, D_, eps_):
    i = t.threadIdx.x
    if i < N_:
        s = 0.0
        sq = 0.0
        for d in range(D_):
            v = t.gload(tmp_base + i * D_ + d)
            s += v
            sq += v * v
        mean = s / D_
        var = sq / D_ - mean * mean
        inv_std = 1.0 / (var + eps_) ** 0.5
        for d in range(D_):
            v = t.gload(tmp_base + i * D_ + d)
            norm = (v - mean) * inv_std
            t.gstore(out_base + i * D_ + d, norm * t.gload(g_base + d) + t.gload(b_base + d))


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(4)
    x = rng.randn(N, D).astype(np.float64)
    residual = rng.randn(N, D).astype(np.float64)
    gamma = rng.randn(D).astype(np.float64)
    beta = rng.randn(D).astype(np.float64)

    v = x + residual
    mean = v.mean(axis=1, keepdims=True)
    var = v.var(axis=1, keepdims=True)
    norm = (v - mean) / np.sqrt(var + EPS)
    ref = norm * gamma + beta

    # Unfused ceiling: measured by actually running a 2-kernel add-then-
    # normalize pipeline through the simulator.
    out_base = 0
    x_base = N * D
    r_base = x_base + N * D
    tmp_base = r_base + N * D
    g_base = tmp_base + N * D
    b_base = g_base + D
    g_unfused = GPU(b_base + D)
    g_unfused.gmem[x_base:x_base + N * D] = x.ravel()
    g_unfused.gmem[r_base:r_base + N * D] = residual.ravel()
    g_unfused.gmem[g_base:g_base + D] = gamma
    g_unfused.gmem[b_base:b_base + D] = beta
    mA = g_unfused.launch(_unfused_add_kernel, 1, N, x_base, r_base, tmp_base, N, D)
    mB = g_unfused.launch(_unfused_layernorm_kernel, 1, N, out_base, tmp_base, g_base, b_base, N, D, EPS)
    unfused_transactions = max(mA["transactions"] + mB["transactions"], 1)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "transaction_ratio": float("inf"), "error": str(e)}

    out_base2 = 0
    x_base2 = N * D
    r_base2 = x_base2 + N * D
    g_base2 = r_base2 + N * D
    b_base2 = g_base2 + D
    gpu = GPU(b_base2 + D)
    gpu.gmem[out_base2:out_base2 + N * D] = 0.0
    gpu.gmem[x_base2:x_base2 + N * D] = x.ravel()
    gpu.gmem[r_base2:r_base2 + N * D] = residual.ravel()
    gpu.gmem[g_base2:g_base2 + D] = gamma
    gpu.gmem[b_base2:b_base2 + D] = beta

    params = {"out": out_base2, "x": x_base2, "residual": r_base2, "gamma": g_base2,
              "beta": b_base2, "N": N, "D": D, "eps": EPS}
    try:
        m = prog.launch(gpu, "fused_layernorm_residual", 1, N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "transaction_ratio": float("inf"), "error": str(e)}

    out = gpu.gmem[out_base2:out_base2 + N * D].reshape(N, D)
    max_err = float(np.max(np.abs(out - ref)))
    return {"max_abs_err": max_err, "transaction_ratio": float(m["transactions"]) / unfused_transactions}


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
