"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

The oracle rebuilds the expected per-lane fragment layout independently in
Python (the same groupID/threadID_in_group formula documented in
sol/ref.cu, not read from the kernel).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M, K, N = 16, 16, 8  # A: M x K, B: K x N (m16n8k16)


def _reference(A, B):
    fragA = np.zeros(32 * 8)
    fragB = np.zeros(32 * 4)
    for lane in range(32):
        group = lane // 4
        tid_in_group = lane % 4
        for k in range(8):
            half = k // 4
            sub = k % 4
            row = group + half * 8
            col = tid_in_group * 4 + sub
            fragA[lane * 8 + k] = A[row, col]
        for k in range(4):
            row = tid_in_group * 4 + k
            col = group
            fragB[lane * 4 + k] = B[row, col]
    return fragA, fragB


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(21)
    A = rng.uniform(-1.0, 1.0, size=(M, K))
    B = rng.uniform(-1.0, 1.0, size=(K, N))
    fragA_ref, fragB_ref = _reference(A, B)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    a_size, b_size, fa_size, fb_size = M * K, K * N, 32 * 8, 32 * 4
    gpu = GPU(fa_size + fb_size + a_size + b_size)
    gpu.gmem[0:fa_size] = 0.0
    gpu.gmem[fa_size:fa_size + fb_size] = 0.0
    gpu.gmem[fa_size + fb_size:fa_size + fb_size + a_size] = A.flatten()
    gpu.gmem[fa_size + fb_size + a_size:fa_size + fb_size + a_size + b_size] = B.flatten()

    params = {
        "fragA_out": 0,
        "fragB_out": fa_size,
        "A": fa_size + fb_size,
        "B": fa_size + fb_size + a_size,
    }
    try:
        prog.launch(gpu, "pack_mma_fragment", 1, 32, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    fragA_got = gpu.gmem[0:fa_size]
    fragB_got = gpu.gmem[fa_size:fa_size + fb_size]
    err = max(float(np.max(np.abs(fragA_got - fragA_ref))),
              float(np.max(np.abs(fragB_got - fragB_ref))))
    return {"max_abs_err": err}


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
