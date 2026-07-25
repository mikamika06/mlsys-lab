"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the read-back values against a numpy oracle and reports the simulator's
smem_waves -- the true, measured n-way bank-conflict degree of the fixed
`idx` access pattern (broadcast reads to an identical address are exempt,
since GPU.launch's per-bank accounting de-duplicates identical addresses).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK, SMEM_WORDS = 32, 32, 128

OUT_BASE = 0
SEED_BASE = N
IDX_BASE = SEED_BASE + SMEM_WORDS
GMEM_SIZE = IDX_BASE + N


def _fixed_idx():
    # lanes 0-15: distinct banks 0-15, one lane each -> no conflict.
    # lanes 16-27: all read word 16 (bank 16) -- a BROADCAST, exempt.
    # lanes 28-31: words 17, 49, 81, 113 -- all bank 17 (17 mod 32), four
    #   DISTINCT addresses -> a genuine 4-way conflict.
    idx = np.zeros(N, dtype=np.float64)
    for lane in range(16):
        idx[lane] = lane
    for lane in range(16, 28):
        idx[lane] = 16
    idx[28], idx[29], idx[30], idx[31] = 17, 49, 81, 113
    return idx


def grade(srcfile: str = "solve.cu") -> dict:
    idx = _fixed_idx()
    seed = 100.0 + 3.0 * np.arange(SMEM_WORDS, dtype=np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "smem_waves": 10 ** 9, "error": str(e)}

    gpu = GPU(GMEM_SIZE, smem_size=SMEM_WORDS)
    gpu.gmem[OUT_BASE:OUT_BASE + N] = -1.0
    gpu.gmem[SEED_BASE:SEED_BASE + SMEM_WORDS] = seed
    gpu.gmem[IDX_BASE:IDX_BASE + N] = idx

    params = {"out": OUT_BASE, "seed": SEED_BASE, "idx": IDX_BASE}
    try:
        m = prog.launch(gpu, "bank_conflict_probe", 1, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "smem_waves": 10 ** 9, "error": str(e)}

    ref_out = seed[idx.astype(np.int64)]
    max_err = float(np.max(np.abs(gpu.gmem[OUT_BASE:OUT_BASE + N] - ref_out)))
    return {"max_abs_err": max_err, "smem_waves": int(m["smem_waves"])}


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
