"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the broadcast-add result against a numpy oracle and reports the transaction
count the simulator observed (coalesced access is the whole point of the
row/col index derivation).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS, COLS, BLOCK = 4, 64, 64
N = ROWS * COLS


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(11)
    mat = rng.randn(ROWS, COLS).astype(np.float64)
    vec = rng.randn(COLS).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    gpu = GPU(2 * N + COLS)
    gpu.gmem[0:N] = 0.0                       # out = gmem[0:N]
    gpu.gmem[N:2 * N] = mat.reshape(-1)       # mat = gmem[N:2N]
    gpu.gmem[2 * N:2 * N + COLS] = vec        # vec = gmem[2N:2N+COLS]

    params = {"out": 0, "mat": N, "vec": 2 * N, "rows": ROWS, "cols": COLS}
    try:
        m = prog.launch(gpu, "broadcast_add", N // BLOCK, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    ref_out = mat + vec[None, :]
    max_err = float(np.max(np.abs(gpu.gmem[0:N] - ref_out.reshape(-1))))
    return {"max_abs_err": max_err, "transactions": int(m["transactions"])}


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
