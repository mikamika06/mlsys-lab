"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the softmax output against a numpy oracle on a fixture with one very
large logit -- unsafe (no max-subtraction) exponentiation would overflow
to inf on this input.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 10, 32

OUT_BASE = 0
X_BASE = N
GMEM_SIZE = X_BASE + N


def _fixture():
    rng = np.random.RandomState(11)
    # a logit of 750 makes exp(750) itself overflow float64 -- max
    # subtraction is not optional here, it's the only thing that keeps
    # this finite.
    return np.concatenate([rng.uniform(0, 5, size=9), [750.0]]).astype(np.float64)


def grade(srcfile: str = "solve.cu") -> dict:
    x = _fixture()

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[OUT_BASE:OUT_BASE + N] = -1.0
    gpu.gmem[X_BASE:X_BASE + N] = x

    params = {"out": OUT_BASE, "x": X_BASE, "n": N}
    try:
        prog.launch(gpu, "safe_softmax", 1, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    exps = np.exp(x - x.max())
    ref_out = exps / exps.sum()
    out = gpu.gmem[OUT_BASE:OUT_BASE + N]
    if not np.all(np.isfinite(out)):
        return {"max_abs_err": float("inf")}
    max_err = float(np.max(np.abs(out - ref_out)))
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
