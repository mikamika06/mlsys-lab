"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).
Checks the modeled peak-memory ratio between materialized attention
scores and FlashAttention's block-at-a-time computation, against a
reference computed directly with numpy (never hardcoded per-scenario).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def grade(srcfile: str = "solve.cu") -> dict:
    seq_len = np.array([128, 512, 2048, 4096, 1024], dtype=np.float64)
    block_size = np.array([32, 64, 64, 128, 32], dtype=np.float64)
    bytes_per_elem = np.array([4, 4, 2, 2, 4], dtype=np.float64)
    n = len(seq_len)
    ref = (seq_len ** 2 * bytes_per_elem) / (block_size ** 2 * bytes_per_elem)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    out_base = 0
    s_base = n
    b_base = 2 * n
    e_base = 3 * n
    gpu = GPU(4 * n)
    gpu.gmem[out_base:out_base + n] = 0.0
    gpu.gmem[s_base:s_base + n] = seq_len
    gpu.gmem[b_base:b_base + n] = block_size
    gpu.gmem[e_base:e_base + n] = bytes_per_elem

    params = {"out": out_base, "seq_len": s_base, "block_size": b_base,
              "bytes_per_elem": e_base, "n": n}
    try:
        prog.launch(gpu, "flash_vs_materialized_ratio", 1, n, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    out = gpu.gmem[out_base:out_base + n]
    max_err = float(np.max(np.abs(out - ref)))
    return {"max_abs_err": max_err}


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
