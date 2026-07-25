"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
the resulting gmem against a numpy reference and reports the transaction
count the simulator observed -- an SoA-vs-AoS-layout gather is either
coalesced or badly scattered depending on which axis the loop strides over.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D, V, N, BLOCK = 32, 256, 32, 32

EMB_BASE = 0
IDX_BASE = D * V
OUT_BASE = IDX_BASE + N
GMEM_SIZE = OUT_BASE + N


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(7)
    emb2d = rng.randn(D, V).astype(np.float64)      # emb2d[d, v]
    idx = np.arange(N, dtype=np.int64)               # thread i gathers row i

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[EMB_BASE:EMB_BASE + D * V] = emb2d.reshape(-1)  # SoA: emb[d*V + v]
    gpu.gmem[IDX_BASE:IDX_BASE + N] = idx.astype(np.float64)
    gpu.gmem[OUT_BASE:OUT_BASE + N] = 0.0

    params = {"out": OUT_BASE, "emb": EMB_BASE, "idx": IDX_BASE, "D": D, "V": V}
    try:
        m = prog.launch(gpu, "gather_soa", N // BLOCK, BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    ref_out = emb2d[:, idx].sum(axis=0)  # sum over d of emb2d[d, idx[i]]
    max_err = float(np.max(np.abs(gpu.gmem[OUT_BASE:OUT_BASE + N] - ref_out)))
    return {"max_abs_err": max_err, "transactions": int(m["transactions"])}


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
