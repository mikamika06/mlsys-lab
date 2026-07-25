"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
a per-row-quantized int8 matvec against a numpy oracle and reports both
`transactions` and `cycles` -- a kernel that dequantizes each weight
in-register right where it's consumed, and caches the activation vector in
shared memory instead of re-reading it from global once per row, touches
global memory far less than one that doesn't.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M, N = 8, 8  # output rows (quantization channels), reduction dim
NW, NS, NX, NY = M * N, M, N, M  # weight ints, per-row scales, activations, outputs


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(11)
    w_int = rng.randint(-127, 127, size=NW).astype(np.float64)
    scale = (0.01 + 0.02 * rng.rand(NS)).astype(np.float64)
    x = rng.randn(NX).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "cycles": 10 ** 9, "error": str(e)}

    gpu = GPU(NW + NS + NX + NY, smem_size=8)
    gpu.gmem[0:NW] = w_int                      # w_int  = gmem[0:NW],           shape (M,N)
    gpu.gmem[NW:NW + NS] = scale                # scale  = gmem[NW:NW+NS],       one per row
    gpu.gmem[NW + NS:NW + NS + NX] = x          # x      = gmem[NW+NS:NW+NS+NX]
    gpu.gmem[NW + NS + NX:NW + NS + NX + NY] = 0.0  # y  = gmem[..:..+NY]

    params = {"y": NW + NS + NX, "w_int": 0, "scale": NW, "x": NW + NS, "M": M, "N": N}
    try:
        m = prog.launch(gpu, "dequant_matvec", 1, M, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "cycles": 10 ** 9, "error": str(e)}

    # Reference: dequantize per-row (w_int[i,j] * scale[i]) then matvec by x.
    ref_out = (w_int.reshape(M, N) * scale.reshape(M, 1)) @ x
    max_err = float(np.max(np.abs(gpu.gmem[NW + NS + NX:NW + NS + NX + NY] - ref_out)))
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
