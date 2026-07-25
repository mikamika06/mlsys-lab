"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the resulting Q@K^T tile against a numpy oracle and reports both the global
memory `transactions` count and the total `cycles` estimate -- one FlashAttention
tile step's whole story is how many bytes get pulled from slow global memory
ONCE into shared memory versus how many times the same bytes get redundantly
re-fetched from global memory instead of reused from the fast tile.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BQ, BK, D = 8, 8, 8   # query tile rows, key tile rows, head dim
NQ, NK, NOUT = BQ * D, BK * D, BQ * BK  # 64, 64, 64


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(3)
    Q = rng.randn(NQ).astype(np.float64)
    K = rng.randn(NK).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "cycles": 10 ** 9, "error": str(e)}

    gpu = GPU(NQ + NK + NOUT, smem_size=NQ + NK)
    gpu.gmem[0:NQ] = Q                    # Q tile = gmem[0:NQ]
    gpu.gmem[NQ:NQ + NK] = K              # K tile = gmem[NQ:NQ+NK]
    gpu.gmem[NQ + NK:NQ + NK + NOUT] = 0.0  # out    = gmem[NQ+NK:NQ+NK+NOUT]

    params = {"out": NQ + NK, "Q": 0, "K": NQ, "BQ": BQ, "BK": BK, "D": D}
    try:
        m = prog.launch(gpu, "qk_tile", 1, BQ * BK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "cycles": 10 ** 9, "error": str(e)}

    ref_out = (Q.reshape(BQ, D) @ K.reshape(BK, D).T).reshape(-1)
    max_err = float(np.max(np.abs(gpu.gmem[NQ + NK:NQ + NK + NOUT] - ref_out)))
    return {"max_abs_err": max_err, "transactions": int(m["transactions"]), "cycles": int(m["cycles"])}


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
