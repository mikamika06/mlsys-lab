"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the resulting gmem against a numpy reference transpose and reports the
transaction count the simulator observed (a naive per-thread transpose is
correct but coalesces on only one side; a shared-memory-tiled one coalesces
both the read and the write).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
B, H, S, D = 2, 2, 16, 16
TOTAL = B * H * S * D  # 1024
TILE_WORDS = S * D     # 256 -- one (b,h) slice's shared-memory tile


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(7)
    x = rng.randn(TOTAL).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    gpu = GPU(2 * TOTAL, smem_size=TILE_WORDS)
    gpu.gmem[0:TOTAL] = x           # in  = gmem[0:TOTAL], shape (B,H,S,D)
    gpu.gmem[TOTAL:2 * TOTAL] = 0.0  # out = gmem[TOTAL:2*TOTAL], shape (B,H,D,S)

    params = {"out": TOTAL, "in": 0, "B": B, "H": H, "S": S, "D": D}
    try:
        m = prog.launch(gpu, "transpose_bhsd", B * H, S * D, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    ref_out = x.reshape(B, H, S, D).transpose(0, 1, 3, 2).reshape(-1)
    max_err = float(np.max(np.abs(gpu.gmem[TOTAL:2 * TOTAL] - ref_out)))
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
