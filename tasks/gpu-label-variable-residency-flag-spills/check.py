"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the label and spill vectors against a numpy oracle that walks the same
running-budget rule.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 10
BUDGET = 8.0

LABEL_BASE = 0
SPILL_BASE = N
KIND_BASE = SPILL_BASE + N
SIZE_BASE = KIND_BASE + N
GMEM_SIZE = SIZE_BASE + N


def _fixture():
    kind = np.array([0, 1, 0, 2, 0, 3, 0, 0, 0, 1], dtype=np.float64)
    size = np.array([2, 16, 3, 1000, 2, 1, 4, 1, 5, 8], dtype=np.float64)
    return kind, size


def _oracle(kind, size, budget):
    label = np.zeros(len(kind))
    spill = np.zeros(len(kind))
    running = 0.0
    for i in range(len(kind)):
        k = kind[i]
        if k == 0.0:
            if running + size[i] <= budget:
                label[i] = 0.0
                running += size[i]
            else:
                label[i] = 4.0
                spill[i] = 1.0
        else:
            label[i] = k
    return label, spill


def grade(srcfile: str = "solve.cu") -> dict:
    kind, size = _fixture()

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[LABEL_BASE:LABEL_BASE + N] = -1.0
    gpu.gmem[SPILL_BASE:SPILL_BASE + N] = -1.0
    gpu.gmem[KIND_BASE:KIND_BASE + N] = kind
    gpu.gmem[SIZE_BASE:SIZE_BASE + N] = size

    params = {"label": LABEL_BASE, "spill": SPILL_BASE, "kind": KIND_BASE,
              "size": SIZE_BASE, "budget": BUDGET, "n": N}
    try:
        prog.launch(gpu, "classify_residency", 1, 32, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref_label, ref_spill = _oracle(kind, size, BUDGET)
    label_err = np.max(np.abs(gpu.gmem[LABEL_BASE:LABEL_BASE + N] - ref_label))
    spill_err = np.max(np.abs(gpu.gmem[SPILL_BASE:SPILL_BASE + N] - ref_spill))
    return {"max_abs_err": float(max(label_err, spill_err))}


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
