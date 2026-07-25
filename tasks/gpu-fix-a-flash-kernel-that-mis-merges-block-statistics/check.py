"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Checks the online-softmax attention output against a standard (unblocked)
softmax-attention numpy oracle -- mathematically identical to a CORRECTLY
merged blocked computation, so any mismatch is a real merge bug, not an
approximation.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NUM_Q, D, NUM_K, BLOCK_K = 8, 4, 16, 8


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(3)
    Q = rng.randn(NUM_Q, D).astype(np.float64) * 1.5
    # Second K-block deliberately larger in magnitude than the first, so
    # the running max genuinely increases partway through -- alpha != 1
    # on that merge, exposing a missing-rescale bug.
    K = np.vstack([rng.randn(8, D) * 1.0, rng.randn(8, D) * 3.0]).astype(np.float64)
    V = rng.randn(NUM_K, D).astype(np.float64)

    scores = Q @ K.T
    scores = scores - scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(axis=1, keepdims=True)
    ref_O = p @ V

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    o_base = 0
    q_base = NUM_Q * D
    k_base = q_base + NUM_Q * D
    v_base = k_base + NUM_K * D
    gpu = GPU(v_base + NUM_K * D)
    gpu.gmem[o_base:o_base + NUM_Q * D] = 0.0
    gpu.gmem[q_base:q_base + NUM_Q * D] = Q.ravel()
    gpu.gmem[k_base:k_base + NUM_K * D] = K.ravel()
    gpu.gmem[v_base:v_base + NUM_K * D] = V.ravel()

    params = {"O": o_base, "Q": q_base, "K": k_base, "V": v_base,
              "num_queries": NUM_Q, "D": D, "num_keys": NUM_K, "block_k": BLOCK_K}
    try:
        prog.launch(gpu, "flash_attn_row", 1, NUM_Q, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    O = gpu.gmem[o_base:o_base + NUM_Q * D].reshape(NUM_Q, D)
    max_err = float(np.max(np.abs(O - ref_O)))
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
