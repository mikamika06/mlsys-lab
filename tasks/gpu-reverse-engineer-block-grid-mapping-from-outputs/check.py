"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU), once per observed
output array, comparing the recovered (mapping_kind, gridDim) against an
independently computed oracle.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 64  # blockDim is always 32; N=64 so gridDim=2 for the flat/2d launches


def _make_flat():
    # 2 blocks of 32, blockDim*gridDim == N: thread t writes out[t] = t.
    return np.arange(N, dtype=float), 0.0, 2.0


def _make_stride():
    # 1 block of 32 (gridDim=1), a grid-stride loop with stride 32 covers
    # all 64 outputs: out[t] is the id of whichever thread wrote it, t % 32.
    return np.array([t % 32 for t in range(N)], dtype=float), 1.0, 1.0


def _make_2d():
    # 2 blocks of 32 again (gridDim=2), but a transposed 8x8 index order.
    return np.array([(t % 8) * 8 + (t // 8) for t in range(N)], dtype=float), 2.0, 2.0


SCENARIOS = [_make_flat(), _make_stride(), _make_2d()]


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"exact_match": 0.0, "error": str(e)}

    for obs, kind_ref, grid_ref in SCENARIOS:
        gpu = GPU(N + 2)
        gpu.gmem[0:N] = obs
        gpu.gmem[N:N + 2] = 0.0
        params = {"obs": 0, "n": N, "result": N}
        try:
            prog.launch(gpu, "reconstruct_launch", 1, 1, params)
        except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
            return {"exact_match": 0.0, "error": str(e)}
        if float(gpu.gmem[N]) != kind_ref or float(gpu.gmem[N + 1]) != grid_ref:
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}


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
