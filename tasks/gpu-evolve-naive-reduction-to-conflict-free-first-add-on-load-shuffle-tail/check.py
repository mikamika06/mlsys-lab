"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Checks
correctness against a numpy sum AND the simulator's observed smem_waves /
cycles -- Harris's optimization ladder (first-add-on-load, conflict-free
addressing, warp-shuffle tail) should cut both drastically versus the naive
kernel1-style reduction.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK, SMEM = 256, 256, 256


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(1)
    x = rng.randn(N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "smem_waves": 10 ** 9, "cycles": 10 ** 9, "error": str(e)}

    gpu = GPU(N + 1, smem_size=SMEM)
    gpu.gmem[0] = 0.0
    gpu.gmem[1:1 + N] = x

    params = {"out": 0, "x": 1, "n": N}
    try:
        m = prog.launch(gpu, "reduce_sum", 1, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "smem_waves": 10 ** 9, "cycles": 10 ** 9, "error": str(e)}

    max_err = float(abs(gpu.gmem[0] - float(x.sum())))
    return {"max_abs_err": max_err, "smem_waves": int(m["smem_waves"]), "cycles": int(m["cycles"])}


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
