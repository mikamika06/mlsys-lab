"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). The
kernel evaluates a real data-dependent predicate per element; this grader
then aggregates those REAL per-thread results into per-warp path counts (1
if a warp's 32 lanes all agree, 2 if they split) and compares against an
oracle built the same way from numpy's own independent evaluation of the
predicate.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK, WARP = 128, 32, 32
THRESHOLD = 0.0


def _build_x():
    rng = np.random.RandomState(8)
    x = np.zeros(N)
    x[0:32] = rng.uniform(1.0, 5.0, size=32)        # warp 0: all positive -> uniform
    x[32:64] = rng.uniform(-5.0, -1.0, size=32)      # warp 1: all negative -> uniform
    half_a = rng.uniform(1.0, 5.0, size=16)
    half_b = rng.uniform(-5.0, -1.0, size=16)
    x[64:96] = np.where(np.arange(32) % 2 == 0,
                         np.repeat(half_a, 2)[:32], np.repeat(half_b, 2)[:32])  # warp 2: mixed
    x[96:128] = rng.uniform(1.0, 5.0, size=32)
    x[96] = -3.0                                      # warp 3: mostly positive, 1 negative
    return x


def _per_warp_paths(pred):
    out = []
    for w in range(0, len(pred), WARP):
        seg = pred[w:w + WARP]
        out.append(1 if len(set(seg.tolist())) == 1 else 2)
    return out


def grade(srcfile: str = "solve.cu") -> dict:
    x = _build_x()

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()
    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"exact_match": 0.0, "error": str(e)}

    gpu = GPU(2 * N)
    gpu.gmem[0:N] = x
    gpu.gmem[N:2 * N] = 0.0
    params = {"x": 0, "pred_out": N, "n": N, "threshold": THRESHOLD}

    try:
        grid = (N + BLOCK - 1) // BLOCK
        prog.launch(gpu, "eval_predicate", grid, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"exact_match": 0.0, "error": str(e)}

    pred = gpu.gmem[N:2 * N]
    got = _per_warp_paths(pred)
    ref = _per_warp_paths((x > THRESHOLD).astype(float))
    return {"exact_match": 1.0 if got == ref else 0.0}


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
