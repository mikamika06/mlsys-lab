"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU). Launches the SAME
kernel twice against the same 200-value fixture, once with a clamp at
fp16's max finite magnitude (modeling fp16 saturating overflow) and once
with an effectively unbounded clamp (modeling fp32, which never gets close
to its own, vastly larger, ceiling on this fixture).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 200
FP16_MAX = 65504.0
NO_CLAMP = 1.0e30  # far above anything this fixture's true sum can reach


def _run(prog, x, clamp_max):
    gpu = GPU(N + 1)
    gpu.gmem[0:N] = x
    gpu.gmem[N] = 0.0
    params = {"x": 0, "out": N, "n": N, "clamp_max": clamp_max}
    prog.launch(gpu, "accumulate_clamped", 1, 1, params)
    return float(gpu.gmem[N])


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(2)
    x = rng.uniform(300.0, 700.0, size=N)
    true_sum = float(np.sum(x))

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()
    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"fp32_err": float("inf"), "fp16_err": 0.0, "error": str(e)}

    try:
        fp16_result = _run(prog, x, FP16_MAX)
        fp32_result = _run(prog, x, NO_CLAMP)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"fp32_err": float("inf"), "fp16_err": 0.0, "error": str(e)}

    return {
        "fp32_err": abs(fp32_result - true_sum),
        "fp16_err": abs(fp16_result - true_sum),
    }


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
