"""Grade a REAL CUDA-C solve.cu containing TWO kernels: parse it once with
arena.cuda_c.CudaProgram and launch each kernel separately (fresh GPU each
time) on the software GPU. Both must compute the same inclusive scan
(checked against a numpy oracle); the point is the measured ARITHMETIC-OP
count (index math and all -- exactly what a real profiler's ALU-op counter
would show), which the simulator folds into `cycles` alongside memory
transactions. Back it out: alu_ops = cycles - transactions*CYC_MEM -
smem_waves*CYC_SMEM (CYC_ALU == 1).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU, CYC_MEM, CYC_SMEM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 256


def _run(prog, kernel_name, x):
    gpu = GPU(2 * N, smem_size=2 * N)
    gpu.gmem[0:N] = x
    gpu.gmem[N:2 * N] = 0.0
    params = {"out": N, "in": 0, "n": N}
    m = prog.launch(gpu, kernel_name, 1, N, params)
    alu_ops = int(m["cycles"] - m["transactions"] * CYC_MEM - m["smem_waves"] * CYC_SMEM)
    return gpu.gmem[N:2 * N], alu_ops


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(9)
    x = rng.uniform(-4.0, 4.0, size=N).astype(np.float64)
    oracle = np.cumsum(x)  # inclusive scan

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "ops_hillis": 10 ** 9,
                 "ops_blelloch": 10 ** 9, "error": str(e)}

    try:
        out_hs, ops_hs = _run(prog, "hillis_steele_scan", x)
        out_bl, ops_bl = _run(prog, "blelloch_scan", x)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "ops_hillis": 10 ** 9,
                 "ops_blelloch": 10 ** 9, "error": str(e)}

    max_err = float(max(np.max(np.abs(out_hs - oracle)), np.max(np.abs(out_bl - oracle))))
    return {"max_abs_err": max_err, "ops_hillis": ops_hs, "ops_blelloch": ops_bl}


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
