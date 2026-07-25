"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

The fixture is one huge value plus 127 small ones. A correctly *ordered*
(balanced-tree) reduction keeps far more precision than a naive
fully-sequential accumulation would, because the small values get combined
with each other -- reaching a magnitude comparable to the huge value's ULP
-- well before they are folded into the huge value's own lineage. The
tolerance below is the ACTUAL error the reference tree reduction achieves on
this fixture (measured, not guessed), with headroom; a fully-sequential
single-thread accumulation on the same fixture loses more than 100x that.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 128, 128
BASE = float(2 ** 54)  # ULP here is 4.0 -- individual ~1-magnitude adds can round away


def _fixture():
    rng = np.random.RandomState(2026)
    small = rng.uniform(0.1, 2.0, size=N - 1)
    x = np.empty(N, dtype=np.float64)
    x[0] = BASE
    x[1:] = small
    return x


def grade(srcfile: str = "solve.cu") -> dict:
    x = _fixture()
    ref_sum = math.fsum(x.tolist())  # correctly-rounded exact sum -- the ground truth

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "smem_waves": 10 ** 9, "error": str(e)}

    gpu = GPU(N + 1, smem_size=BLOCK)
    gpu.gmem[0:N] = x       # in  = gmem[0:N]
    gpu.gmem[N] = 0.0       # out = gmem[N]

    params = {"out": N, "in": 0, "n": N}
    try:
        m = prog.launch(gpu, "reduce_sum", 1, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "smem_waves": 10 ** 9, "error": str(e)}

    gpu_result = float(gpu.gmem[N])
    max_err = abs(gpu_result - ref_sum)
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
