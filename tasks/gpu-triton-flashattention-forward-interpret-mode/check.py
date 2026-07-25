"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Compares against ordinary (materialize-the-full-score-row) softmax
attention computed directly in numpy -- the two must agree exactly, since
online softmax is a mathematically equivalent, numerically stable way to
compute the same quantity, not an approximation of it.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, D, BLOCK = 16, 4, 32


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(31)
    Q = rng.uniform(-2.0, 2.0, size=(N, D))
    K = rng.uniform(-2.0, 2.0, size=(N, D))
    V = rng.uniform(-2.0, 2.0, size=(N, D))
    scale = 1.0 / np.sqrt(D)

    scores = (Q @ K.T) * scale
    scores = scores - scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(axis=1, keepdims=True)
    expected = p @ V
    denom = float(np.max(np.abs(expected)))

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()
    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"rel_err": float("inf"), "error": str(e)}

    gpu = GPU(4 * N * D)
    gpu.gmem[0:N * D] = Q.flatten()
    gpu.gmem[N * D:2 * N * D] = K.flatten()
    gpu.gmem[2 * N * D:3 * N * D] = V.flatten()
    gpu.gmem[3 * N * D:4 * N * D] = 0.0
    params = {"Q": 0, "K": N * D, "V": 2 * N * D, "O": 3 * N * D, "N": N, "scale": float(scale)}

    try:
        grid = (N + BLOCK - 1) // BLOCK
        prog.launch(gpu, "flash_attention_fwd", grid, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"rel_err": float("inf"), "error": str(e)}

    O = gpu.gmem[3 * N * D:4 * N * D].reshape(N, D)
    rel_err = float(np.max(np.abs(O - expected))) / denom
    return {"rel_err": rel_err}


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
