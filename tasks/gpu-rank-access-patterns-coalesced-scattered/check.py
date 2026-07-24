"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute three kernels -- unit_stride, reversed_stride, stride4 -- on the
software GPU (arena.cuda_sim.GPU), each on its own fresh GPU instance so
they can't interfere with each other. Compares each kernel's output against
a numpy oracle and reports each one's REAL measured transaction count.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 64, 32
A = 3.0

KERNELS = {
    "unit_stride": N,        # g[idx], idx in [0, N)          -> needs N slots
    "reversed_stride": N,    # g[n-1-idx], idx in [0, N)       -> needs N slots
    "stride4": 4 * N,        # g[idx*4], idx in [0, N)         -> needs 4N slots
}


def _expected(kname, x, n, a):
    y = x.copy()
    if kname == "unit_stride":
        y[:n] = a * x[:n]
    elif kname == "reversed_stride":
        y[:n] = a * x[:n]  # same SET of elements as unit_stride, just visited
        # in reverse order per-thread -- the final array is identical.
    elif kname == "stride4":
        idx = np.arange(n) * 4
        y[idx] = a * x[idx]
    return y


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {
            "max_abs_err": float("inf"),
            "transactions_unit": 10 ** 9,
            "transactions_reversed": 10 ** 9,
            "transactions_stride4": 0,
            "error": str(e),
        }

    rng = np.random.RandomState(11)
    max_err = 0.0
    txn = {}
    for kname, gsize in KERNELS.items():
        x = rng.randn(gsize).astype(np.float64)
        gpu = GPU(gsize)
        gpu.gmem[:] = x
        params = {"g": 0, "a": A, "n": N}
        try:
            m = prog.launch(gpu, kname, N // BLOCK, BLOCK, params)
        except Exception as e:  # noqa: BLE001 -- any runtime fault fails cleanly
            return {
                "max_abs_err": float("inf"),
                "transactions_unit": 10 ** 9,
                "transactions_reversed": 10 ** 9,
                "transactions_stride4": 0,
                "error": f"{kname}: {e}",
            }
        expected = _expected(kname, x, N, A)
        max_err = max(max_err, float(np.max(np.abs(gpu.gmem - expected))))
        key = {"unit_stride": "transactions_unit",
               "reversed_stride": "transactions_reversed",
               "stride4": "transactions_stride4"}[kname]
        txn[key] = int(m["transactions"])

    return {"max_abs_err": max_err, **txn}


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
