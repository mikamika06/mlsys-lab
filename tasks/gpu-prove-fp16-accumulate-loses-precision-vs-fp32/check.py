"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
both accumulators against numpy oracles, and separately confirms the
fixture actually proves the point: the fp32-style accumulator stays close
to the true sum while the fp16-style one is measurably, significantly off.
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
N = 64
BASE = 1024.0


def _fixture():
    rng = np.random.RandomState(21)
    return rng.uniform(-0.4, 0.4, size=N).astype(np.float64)


def _fp16_oracle(inc):
    acc = BASE
    for v in inc:
        acc = math.floor(acc + v + 0.5)
    return acc


def grade(srcfile: str = "solve.cu") -> dict:
    inc = _fixture()

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(2 + 1 + N)
    gpu.gmem[0] = -1.0
    gpu.gmem[1] = -1.0
    gpu.gmem[2] = BASE
    gpu.gmem[3:3 + N] = inc

    params = {"out_fp32": 0, "out_fp16": 1, "base": 2, "inc": 3, "n": N}
    try:
        prog.launch(gpu, "accumulate_precision_demo", 1, 32, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    true_sum = BASE + float(inc.sum())
    ref_fp16 = _fp16_oracle(inc)
    fp32_err = abs(gpu.gmem[0] - true_sum)
    fp16_err = abs(gpu.gmem[1] - ref_fp16)
    return {"max_abs_err": float(max(fp32_err, fp16_err))}


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
