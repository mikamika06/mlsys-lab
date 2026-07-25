"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 64, 32  # grid = 2 blocks of 32 -> exactly N threads, no tail


def _reference(x):
    silu = x / (1.0 + np.exp(-x))
    inner = 0.7978845608 * (x + 0.044715 * x ** 3)
    gelu = 0.5 * x * (1.0 + np.tanh(inner))
    return gelu, silu


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(3)
    x = rng.uniform(-4.0, 4.0, size=N)
    gelu_ref, silu_ref = _reference(x)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(3 * N)
    gpu.gmem[0:N] = 0.0        # gelu_out = gmem[0:N]
    gpu.gmem[N:2 * N] = 0.0    # silu_out = gmem[N:2N]
    gpu.gmem[2 * N:3 * N] = x  # x        = gmem[2N:3N]

    params = {"gelu_out": 0, "silu_out": N, "x": 2 * N, "n": N}
    try:
        prog.launch(gpu, "fused_gelu_silu", N // BLOCK, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    gelu_got = gpu.gmem[0:N]
    silu_got = gpu.gmem[N:2 * N]
    err = max(float(np.max(np.abs(gelu_got - gelu_ref))),
              float(np.max(np.abs(silu_got - silu_ref))))
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
