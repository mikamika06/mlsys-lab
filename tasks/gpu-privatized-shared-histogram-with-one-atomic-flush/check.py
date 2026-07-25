"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

The simulator runs threads one at a time, so even a non-atomic
read-modify-write would happen to sum to the right total here -- the
concept this task is actually about (privatized shared accumulation, one
atomic flush per bin) is graded on the `races` metric the simulator reports
whenever more than one thread writes the same address and at least one of
those writes is non-atomic. A correct privatized-histogram kernel has
races == 0; a naive non-atomic accumulate (shared or global) does not.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 128       # input elements
NBINS = 8
BLOCK = 32
GRID = N // BLOCK  # 4 blocks -> each block's threads collide on bins, and
                    # every block's bin-0..7 flush targets the same 8
                    # global addresses as every other block's flush.


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(42)
    x = rng.randint(0, NBINS, size=N).astype(np.float64)
    ref_hist = np.bincount(x.astype(np.int64), minlength=NBINS).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "races": 10 ** 9, "error": str(e)}

    gpu = GPU(N + NBINS, smem_size=NBINS)
    gpu.gmem[0:N] = x
    gpu.gmem[N:N + NBINS] = 0.0

    params = {"input": 0, "out": N, "n": N}
    try:
        m = prog.launch(gpu, "histogram_privatized", GRID, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "races": 10 ** 9, "error": str(e)}

    got = gpu.gmem[N:N + NBINS]
    max_err = float(np.max(np.abs(got - ref_hist)))
    return {"max_abs_err": max_err, "races": int(m["races"])}


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
