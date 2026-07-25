"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Checks the row-wise softmax against a numpy oracle on a fixture with
LARGE logits (up to +-1000): naive exp(logit) overflows there (this
simulator's expf even raises on it, just like real hardware's expf would
saturate to +inf), so only the numerically safe (max-subtracted) version
can pass.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_ROWS, D = 4, 6


def grade(srcfile: str = "solve.cu") -> dict:
    logits = np.array([
        [1000.0, 1001.0, 999.0, 1000.5, 998.0, 1002.0],
        [50.0, 51.0, 49.0, 50.5, 48.0, 52.0],
        [-1000.0, -999.0, -1001.0, -998.0, -1002.0, -997.0],
        [0.0, 1.0, 2.0, -1.0, 3.0, 0.5],
    ], dtype=np.float64)

    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    ref = e / e.sum(axis=1, keepdims=True)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    out_base = 0
    l_base = N_ROWS * D
    gpu = GPU(l_base + N_ROWS * D)
    gpu.gmem[out_base:out_base + N_ROWS * D] = 0.0
    gpu.gmem[l_base:l_base + N_ROWS * D] = logits.ravel()

    params = {"out": out_base, "logits": l_base, "n_rows": N_ROWS, "D": D}
    try:
        prog.launch(gpu, "safe_softmax_row", 1, N_ROWS, params)
    except Exception as e:  # noqa: BLE001 — overflow/any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    out = gpu.gmem[out_base:out_base + N_ROWS * D].reshape(N_ROWS, D)
    if not np.all(np.isfinite(out)):
        return {"max_abs_err": float("inf"), "error": "non-finite output"}
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
