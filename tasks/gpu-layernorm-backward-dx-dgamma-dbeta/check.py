"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
LayerNorm's backward pass (dx, dgamma, dbeta) against numpy's closed-form
analytic gradient of the forward normalization.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
B, D = 4, 8  # rows (batch), features per row
EPS = 1e-5


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(17)
    x = rng.randn(B * D).astype(np.float64)
    dy = rng.randn(B * D).astype(np.float64)
    gamma = (0.5 + rng.rand(D)).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    layout = [x, dy, gamma, np.zeros(B * D), np.zeros(D), np.zeros(D)]
    names = ["x", "dy", "gamma", "dx", "dgamma", "dbeta"]
    off, b = {}, 0
    for n, a in zip(names, layout):
        off[n] = b
        b += len(a)

    gpu = GPU(b, smem_size=1)
    for n, a in zip(names, layout):
        gpu.gmem[off[n]:off[n] + len(a)] = a

    params = {"dx": off["dx"], "dgamma": off["dgamma"], "dbeta": off["dbeta"],
              "dy": off["dy"], "x": off["x"], "gamma": off["gamma"], "B": B, "D": D}
    try:
        prog.launch(gpu, "layernorm_backward", 1, B * D, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    X, DY = x.reshape(B, D), dy.reshape(B, D)
    mean = X.mean(axis=1, keepdims=True)
    var = ((X - mean) ** 2).mean(axis=1, keepdims=True)
    std = np.sqrt(var + EPS)
    xhat = (X - mean) / std
    g = DY * gamma.reshape(1, D)
    mean_g = g.mean(axis=1, keepdims=True)
    mean_g_xhat = (g * xhat).mean(axis=1, keepdims=True)
    ref_dx = (g - mean_g - xhat * mean_g_xhat) / std
    ref_dgamma = (DY * xhat).sum(axis=0)
    ref_dbeta = DY.sum(axis=0)

    got_dx = gpu.gmem[off["dx"]:off["dx"] + B * D].reshape(B, D)
    got_dgamma = gpu.gmem[off["dgamma"]:off["dgamma"] + D]
    got_dbeta = gpu.gmem[off["dbeta"]:off["dbeta"] + D]

    max_err = max(
        float(np.max(np.abs(got_dx - ref_dx))),
        float(np.max(np.abs(got_dgamma - ref_dgamma))),
        float(np.max(np.abs(got_dbeta - ref_dbeta))),
    )
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
