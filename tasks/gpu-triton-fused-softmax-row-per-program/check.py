"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

Rows are deliberately large-magnitude (600..750) -- softmax is shift-
invariant (softmax(x) == softmax(x - c) for any constant c), so the exact
values don't matter, only their spread, but exp() of an UNSHIFTED value in
that range overflows double precision to +inf, and inf/inf is nan. A
correctly max-shifted kernel is unaffected by the scale at all.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS, COLS = 20, 37


def _reference(x):
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=1, keepdims=True)


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(61)
    x = rng.uniform(600.0, 750.0, size=(ROWS, COLS))
    ref = _reference(x).flatten()

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    n = ROWS * COLS
    gpu = GPU(2 * n)
    gpu.gmem[0:n] = 0.0
    gpu.gmem[n:2 * n] = x.flatten()

    params = {"out": 0, "x": n, "rows": ROWS, "cols": COLS}
    try:
        prog.launch(gpu, "row_softmax", ROWS, 1, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    got = gpu.gmem[0:n]
    diff = got - ref
    max_err = float(np.max(np.abs(diff))) if np.all(np.isfinite(diff)) else float("inf")
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
