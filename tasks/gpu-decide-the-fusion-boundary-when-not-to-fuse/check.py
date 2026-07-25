"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Each
thread decides, for one op-graph edge, whether cutting (materializing the
intermediate) or fusing (recomputing it per consumer) minimizes traffic --
compares the resulting gmem cut-vector against a numpy oracle.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NUM_CASES, MAX_EDGES = 24, 6
N = NUM_CASES * MAX_EDGES
BLOCK = 32


def _make_cases():
    """Deterministic (no rng module, just an arithmetic mix) per-edge
    (size, reuse-count, recompute-cost) triples, with a subset forced close
    to the cut/fuse tie point so the boundary condition is actually
    exercised."""
    size = np.zeros(N, dtype=np.int64)
    reuse = np.zeros(N, dtype=np.int64)
    recompute = np.zeros(N, dtype=np.int64)
    idx = 0
    for c in range(NUM_CASES):
        for e in range(MAX_EDGES):
            s = 8 + ((17 * c + 11 * e + 5) % 97)
            u = 1 + ((5 * c + 7 * e + 3) % 5)
            r = 1 + ((3 * c * c + 13 * e + c * e + 19) % 80)
            if (c + 2 * e) % 11 == 0:
                u = 2 + ((c + e) % 4)
                r = (2 * s) // (u - 1)  # lands exactly on/near the tie point
            size[idx], reuse[idx], recompute[idx] = s, u, r
            idx += 1
    return size, reuse, recompute


def grade(srcfile: str = "solve.cu") -> dict:
    size, reuse, recompute = _make_cases()
    cut_cost = 2 * size
    fuse_cost = recompute * (reuse - 1)
    expected = (cut_cost <= fuse_cost).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"exact_match": 0.0, "error": str(e)}

    size_base, reuse_base, recompute_base, out_base = 0, N, 2 * N, 3 * N
    gpu = GPU(4 * N)
    gpu.gmem[size_base:size_base + N] = size
    gpu.gmem[reuse_base:reuse_base + N] = reuse
    gpu.gmem[recompute_base:recompute_base + N] = recompute

    params = {
        "size": size_base,
        "reuse": reuse_base,
        "recompute": recompute_base,
        "out": out_base,
        "n": N,
    }
    grid = (N + BLOCK - 1) // BLOCK
    try:
        prog.launch(gpu, "fusion_boundary", grid, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"exact_match": 0.0, "error": str(e)}

    got = gpu.gmem[out_base:out_base + N]
    return {"exact_match": 1.0 if np.array_equal(got, expected) else 0.0}


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
