"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the computed serial and overlapped total cycles against a numpy oracle.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 5, 32

SER_BASE = 0
OVL_BASE = N
T_BASE = OVL_BASE + N
L_BASE = T_BASE + N
C_BASE = L_BASE + N
GMEM_SIZE = C_BASE + N


def grade(srcfile: str = "solve.cu") -> dict:
    T = np.array([10.0, 10.0, 1.0, 50.0, 5.0])
    L = np.array([200.0, 500.0, 300.0, 100.0, 1000.0])
    C = np.array([500.0, 200.0, 100.0, 100.0, 50.0])

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[SER_BASE:SER_BASE + N] = -1.0
    gpu.gmem[OVL_BASE:OVL_BASE + N] = -1.0
    gpu.gmem[T_BASE:T_BASE + N] = T
    gpu.gmem[L_BASE:L_BASE + N] = L
    gpu.gmem[C_BASE:C_BASE + N] = C

    params = {"out_serial": SER_BASE, "out_overlap": OVL_BASE, "T": T_BASE, "L": L_BASE, "C": C_BASE, "n": N}
    try:
        prog.launch(gpu, "pipeline_cycles", 1, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref_serial = T * (L + C)
    ref_overlap = L + (T - 1.0) * np.maximum(L, C) + C
    serial_err = np.max(np.abs(gpu.gmem[SER_BASE:SER_BASE + N] - ref_serial))
    overlap_err = np.max(np.abs(gpu.gmem[OVL_BASE:OVL_BASE + N] - ref_overlap))
    return {"max_abs_err": float(max(serial_err, overlap_err))}


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
