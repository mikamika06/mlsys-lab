"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU), once
per named bank-access case (conflict-free / broadcast / 4-way conflict).
Compares BOTH the resulting values (a numpy oracle) and the simulator's
observed smem_waves for every case against the reference's own numbers.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = (0, 1, 2)
# smem_waves the reference pattern produces for each case (a write step
# plus a read-back step, both through the same idx pattern).
EXPECTED_WAVES = {0: 2, 1: 2, 2: 8}


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(5)
    xin = rng.randn(32).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "waves_exact": 0.0, "error": str(e)}

    max_err = 0.0
    waves_ok = 1.0
    for case_id in CASES:
        gpu = GPU(64, smem_size=128)
        gpu.gmem[0:32] = xin
        gpu.gmem[32:64] = -1.0
        params = {"out": 32, "in": 0, "case_id": case_id, "n": 32}
        try:
            m = prog.launch(gpu, "bank_pattern", 1, 32, params)
        except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
            return {"max_abs_err": float("inf"), "waves_exact": 0.0, "error": str(e)}

        # A broadcast write: every thread writes buf[0], so whichever
        # thread the simulator executes LAST for that phase (thread 31,
        # in this deterministic round-robin simulator) is the value every
        # thread reads back.
        ref_out = xin if case_id != 1 else np.full(32, xin[31])
        err = float(np.max(np.abs(gpu.gmem[32:64] - ref_out)))
        max_err = max(max_err, err)
        if int(m["smem_waves"]) != EXPECTED_WAVES[case_id]:
            waves_ok = 0.0

    return {"max_abs_err": max_err, "waves_exact": waves_ok}


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
