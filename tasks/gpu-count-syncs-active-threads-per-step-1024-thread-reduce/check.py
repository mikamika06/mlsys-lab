"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

Checks correctness against a numpy sum oracle AND that the total number of
shared-memory instructions issued matches the closed-form schedule for a
1024-thread, log2(1024) = 10-step tree reduction:

  - n threads each store once into shared memory (the initial load): n
  - step k (k = 0..9) has active_k = n / 2**(k+1) threads, each doing two
    shared-memory reads and one write (3 ops): sum_k active_k * 3
    == 3 * (n - 1), since sum_k active_k == n - 1 (every tree reduction
    over n leaves does exactly n-1 combines, regardless of shape)
  - thread 0 makes one final shared-memory read to hand sdata[0] to
    global memory: 1

  total = n + 3*(n-1) + 1 = 4*n - 2

For n = 1024 that closed form gives 4*1024 - 2 = 4094, which is also the
number the reference kernel actually measures on the simulator (both are
checked below, so the gate threshold is never a guess).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 1024
EXPECTED_SMEM_INSTS = 4 * N - 2  # closed-form schedule total, see module docstring


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(11)
    x = rng.randn(N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "smem_insts": 10 ** 9, "error": str(e)}

    gpu = GPU(N + 1, smem_size=N)
    gpu.gmem[0:N] = x
    gpu.gmem[N] = 0.0

    params = {"out": N, "in": 0, "n": N}
    try:
        m = prog.launch(gpu, "block_reduce_sum", 1, N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "smem_insts": 10 ** 9, "error": str(e)}

    ref_sum = float(x.sum())
    max_err = float(abs(gpu.gmem[N] - ref_sum))
    return {"max_abs_err": max_err, "smem_insts": int(m["smem_insts"])}


if __name__ == "__main__":
    import json

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
