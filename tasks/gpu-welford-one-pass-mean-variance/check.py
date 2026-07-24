"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Compares the resulting mean/variance against numpy's two-pass reference on
a wide-dynamic-range fixture.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.default_rng(42)
    # wide-dynamic-range: mix small and large values
    x = np.concatenate([
        rng.uniform(1e-5, 1e-3, 50),
        rng.uniform(1e3, 1e5, 50),
    ])
    rng.shuffle(x)
    n = len(x)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(n + 2)
    gpu.gmem[:n] = x.astype(np.float64)
    gpu.gmem[n:n + 2] = 0.0

    params = {"x": 0, "out": n, "n": n}
    try:
        prog.launch(gpu, "welford_kernel", 1, 32, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    got_mean = gpu.gmem[n]
    got_var = gpu.gmem[n + 1]

    ref_mean = float(np.mean(x))
    ref_var = float(np.var(x))  # population variance

    err = max(abs(got_mean - ref_mean), abs(got_var - ref_var))
    return {"max_abs_err": float(err)}


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
