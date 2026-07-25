"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU), once per fixed
(N, block_size, max_concurrent) scenario, each launch a single thread that
computes and writes the tail-waste-free block count. Compared against an
independently-computed numpy/Python integer reference, never hardcoded.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# (N, block_size, max_concurrent) -- a spread of cases including some
# where the natural block count already fits in one wave.
SCENARIOS = [
    (1000, 128, 3),
    (100000, 256, 40),
    (50, 32, 8),
    (777, 64, 5),
]


def _oracle(n, block_size, max_concurrent):
    total_blocks = -(-n // block_size)  # ceil
    if total_blocks > max_concurrent:
        return (total_blocks // max_concurrent) * max_concurrent
    return total_blocks


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    got = []
    try:
        for i, (n, bs, mc) in enumerate(SCENARIOS):
            gpu = GPU(len(SCENARIOS))
            gpu.gmem[:] = 0.0
            params = {"out": 0, "idx": i, "N": n, "block_size": bs, "max_concurrent": mc}
            prog.launch(gpu, "optimal_grid_blocks", 1, 1, params)
            got.append(float(gpu.gmem[i]))
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    oracle = [float(_oracle(*s)) for s in SCENARIOS]
    max_err = float(max(abs(a - b) for a, b in zip(got, oracle)))
    return {"max_abs_err": max_err}


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
