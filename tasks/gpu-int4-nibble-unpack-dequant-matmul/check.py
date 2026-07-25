"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the fused unpack+dequant+matvec output against a numpy oracle that
independently unpacks, dequantizes per group, and dots against x.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M, K, G, BLOCK = 4, 8, 4, 32  # 4 output rows, 8 columns, group size 4 (2 groups/row)

Y_BASE = 0
PW_BASE = M
SC_BASE = PW_BASE + M * (K // 2)
X_BASE = SC_BASE + M * (K // G)
GMEM_SIZE = X_BASE + K


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(2)
    codes = rng.randint(0, 16, size=(M, K)).astype(np.float64)
    packed = codes[:, 0::2] + 16.0 * codes[:, 1::2]  # (M, K/2)
    scale = rng.uniform(0.1, 2.0, size=(M, K // G)).astype(np.float64)
    x = rng.uniform(-1, 1, size=K).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[Y_BASE:Y_BASE + M] = -999.0
    gpu.gmem[PW_BASE:PW_BASE + M * (K // 2)] = packed.reshape(-1)
    gpu.gmem[SC_BASE:SC_BASE + M * (K // G)] = scale.reshape(-1)
    gpu.gmem[X_BASE:X_BASE + K] = x

    params = {"y": Y_BASE, "packed_w": PW_BASE, "scale": SC_BASE, "x": X_BASE, "M": M, "K": K, "G": G}
    try:
        prog.launch(gpu, "dequant_matvec", 1, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    dequant = codes * scale[:, np.arange(K) // G]
    y_ref = (dequant * x[None, :]).sum(axis=1)
    max_err = float(np.max(np.abs(gpu.gmem[Y_BASE:Y_BASE + M] - y_ref)))
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
