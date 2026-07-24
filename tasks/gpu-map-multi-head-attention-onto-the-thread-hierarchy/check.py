"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). The
driver fixes a warp-aligned launch shape (block=32, grid=ceil(N/32)) for
each of 5 attention shapes; the kernel must flatten (b,h,s) into its lane's
token id (or -1 if idle) and the result must be a perfect bijection onto
0..N-1 for every shape.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WARP = 32

CASES = [
    (2, 8, 128, 64),
    (3, 5, 129, 64),
    (1, 12, 197, 80),
    (4, 3, 65, 32),
    (2, 7, 255, 96),
]


def _reference_histogram(batch, heads, seq):
    total = batch * heads * seq
    hist = np.zeros(total, dtype=np.int64)
    for b in range(batch):
        for h in range(heads):
            for s in range(seq):
                token = (b * heads + h) * seq + s
                hist[token] += 1
    return hist


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"exact_match": 0.0, "error": str(e)}

    all_exact = True
    for batch, heads, seq, dim in CASES:
        total_tokens = batch * heads * seq
        block = WARP
        grid = math.ceil(total_tokens / block)
        lanes = grid * block

        gpu = GPU(lanes)
        params = {"out": 0, "batch": batch, "heads": heads, "seq": seq,
                  "dim": dim, "total_tokens": total_tokens}
        try:
            prog.launch(gpu, "map_tokens", grid, block, params)
        except Exception:  # noqa: BLE001 — any runtime fault fails the gate cleanly
            return {"exact_match": 0.0}

        values = gpu.gmem[:lanes]
        hist = np.zeros(total_tokens, dtype=np.int64)
        ok = True
        for raw in values:
            if not np.isfinite(raw):
                ok = False
                continue
            nearest = int(round(raw))
            if float(nearest) != float(raw):
                ok = False
                continue
            if 0 <= nearest < total_tokens:
                hist[nearest] += 1
            elif nearest != -1:
                ok = False

        ref_hist = _reference_histogram(batch, heads, seq)
        if not ok or not np.array_equal(hist, ref_hist):
            all_exact = False

    return {"exact_match": 1.0 if all_exact else 0.0}


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
