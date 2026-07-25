"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). A
single 32x32 tile, one block of 1024 threads (32 warps). Compares the
transposed output against a numpy oracle and reports the shared-memory
bank-conflict metric (smem_waves) the simulator observed -- the whole
point of the +1 padding.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 32          # tile side
BLOCK = N * N   # 1024 threads = 32 warps


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(7)
    x = rng.randn(N, N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "smem_waves": 10 ** 9, "error": str(e)}

    gpu = GPU(2 * N * N, smem_size=1056)
    gpu.gmem[0:N * N] = x.reshape(-1)         # in  = gmem[0 : N*N]
    gpu.gmem[N * N:2 * N * N] = 0.0           # out = gmem[N*N : 2*N*N]

    params = {"out": N * N, "in": 0, "n": N}
    try:
        m = prog.launch(gpu, "transpose_tile", 1, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "smem_waves": 10 ** 9, "error": str(e)}

    ref_out = x.T.reshape(-1)
    got_out = gpu.gmem[N * N:2 * N * N]
    max_err = float(np.max(np.abs(got_out - ref_out)))
    return {"max_abs_err": max_err, "smem_waves": int(m["smem_waves"])}


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
