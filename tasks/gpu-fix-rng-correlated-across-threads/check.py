"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). A
correctly de-correlated per-element stream should keep close to half its
elements (mask ~ Bernoulli(0.5) i.i.d. across n=1000 elements, binomial std
~1.6%) -- a stream that's actually one shared draw broadcast to every thread
collapses to either mean 0.0 or mean 1.0, far outside any reasonable
tolerance around 0.5.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 1000, 32
SEED = 12345


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"mean_dev": 1.0, "error": str(e)}

    gpu = GPU(N)
    gpu.gmem[0:N] = 0.0
    params = {"seed": SEED, "out": 0, "n": N}

    try:
        prog.launch(gpu, "dropout_mask", (N + BLOCK - 1) // BLOCK, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"mean_dev": 1.0, "error": str(e)}

    out = gpu.gmem[0:N]
    mean_dev = float(abs(np.mean(out) - 0.5))
    return {"mean_dev": mean_dev}


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
