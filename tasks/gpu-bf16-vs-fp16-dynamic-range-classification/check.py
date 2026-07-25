"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the resulting boolean-as-float vector against a numpy reference computed from
each format's actual max representable finite magnitude.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 64, 32

# fp16 (5 exponent bits, 10 mantissa bits): max finite = (2 - 2^-10) * 2^15.
FP16_MAX = 65504.0
# bf16 (8 exponent bits, 7 mantissa bits -- same exponent range as fp32):
# max finite = (2 - 2^-7) * 2^127.
BF16_MAX = 3.3895313892515355e38


def _build_values():
    rng = np.random.RandomState(7)
    # Wide dynamic range: random sign, exponent uniform in [-10, 40].
    signs = rng.choice([-1.0, 1.0], size=N - 8)
    exps = rng.uniform(-10.0, 40.0, size=N - 8)
    vals = signs * (10.0 ** exps)
    # Explicit boundary cases (both sides of each format's edge).
    edges = np.array([
        FP16_MAX, FP16_MAX + 1.0, FP16_MAX - 1.0, -FP16_MAX - 1.0,
        BF16_MAX, BF16_MAX * 1.001, 0.0, -0.0,
    ])
    return np.concatenate([vals, edges])


def grade(srcfile: str = "solve.cu") -> dict:
    x = _build_values()

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"exact_match": 0.0, "error": str(e)}

    gpu = GPU(2 * N)
    gpu.gmem[0:N] = x
    gpu.gmem[N:2 * N] = 0.0

    params = {"x": 0, "out": N, "n": N}
    try:
        prog.launch(gpu, "classify_overflow", (N + BLOCK - 1) // BLOCK, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"exact_match": 0.0, "error": str(e)}

    ax = np.abs(x)
    ref = ((ax > FP16_MAX) & (ax <= BF16_MAX)).astype(np.float64)
    got = gpu.gmem[N:2 * N]
    exact = 1.0 if np.array_equal(got, ref) else 0.0
    return {"exact_match": exact}


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
