"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU). Compares the scattered
output against a numpy stable sort by digit -- the ground truth any correct
histogram+scan+stable-scatter pass must reproduce exactly.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 20
NUM_DIGITS = 4


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(23)
    keys = np.arange(N, dtype=float)
    digits = rng.randint(0, NUM_DIGITS, size=N).astype(float)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()
    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"exact_match": 0.0, "error": str(e)}

    gpu = GPU(3 * N, smem_size=32)
    gpu.gmem[0:N] = keys
    gpu.gmem[N:2 * N] = digits
    gpu.gmem[2 * N:3 * N] = -1.0
    params = {"keys": 0, "digits": N, "out": 2 * N, "n": N, "num_digits": NUM_DIGITS}

    try:
        prog.launch(gpu, "radix_scatter", 1, 1, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"exact_match": 0.0, "error": str(e)}

    out = gpu.gmem[2 * N:3 * N]
    order = sorted(range(N), key=lambda i: digits[i])
    expected = keys[order]
    return {"exact_match": 1.0 if np.array_equal(out, expected) else 0.0}


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
