"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute BOTH of its kernels on the software GPU (arena.cuda_sim.GPU),
comparing against numpy oracles that replicate the exact same sequence of
float64 additions. Also reports the spread across three "atomic-style"
accumulation orders on the same input -- non-associativity of float
addition means summing the same values in a different order gives a
different bit pattern, even though a fixed-structure tree reduction never
does.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 32
BIG = 1e16


def _fixture():
    # one huge value plus the integers 1..31 (true sum = BIG + 496).
    return np.concatenate([[BIG], np.arange(1, N, dtype=np.float64)])


def _tree_oracle(x):
    a = x.astype(np.float64).copy()
    stride = len(a) // 2
    while stride > 0:
        for i in range(stride):
            a[i] = a[i] + a[i + stride]
        stride //= 2
    return float(a[0])


def _ordered_oracle(x, order):
    acc = 0.0
    for k in order.astype(np.int64):
        acc = acc + float(x[k])
    return acc


def _run_tree(prog, x):
    gpu = GPU(N + 1, smem_size=N)
    gpu.gmem[0] = 0.0
    gpu.gmem[1:1 + N] = x
    prog.launch(gpu, "tree_reduce_sum", 1, N, {"out": 0, "x": 1, "n": N})
    return float(gpu.gmem[0])


def _run_ordered(prog, x, order):
    gpu = GPU(1 + N + N, smem_size=1)
    gpu.gmem[0] = 0.0
    gpu.gmem[1:1 + N] = x
    gpu.gmem[1 + N:1 + 2 * N] = order
    prog.launch(gpu, "ordered_reduce_sum", 1, N, {"out": 0, "x": 1, "order": 1 + N, "n": N})
    return float(gpu.gmem[0])


def grade(srcfile: str = "solve.cu") -> dict:
    x = _fixture()
    identity = np.arange(N, dtype=np.float64)
    reverse = identity[::-1].copy()
    shuffled = np.random.RandomState(1).permutation(N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    try:
        tree = _run_tree(prog, x)
        o1 = _run_ordered(prog, x, identity)
        o2 = _run_ordered(prog, x, reverse)
        o3 = _run_ordered(prog, x, shuffled)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    errs = [
        abs(tree - _tree_oracle(x)),
        abs(o1 - _ordered_oracle(x, identity)),
        abs(o2 - _ordered_oracle(x, reverse)),
        abs(o3 - _ordered_oracle(x, shuffled)),
    ]
    return {"max_abs_err": float(max(errs))}


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
