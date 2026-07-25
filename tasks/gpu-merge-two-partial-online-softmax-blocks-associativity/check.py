"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

Each of N rows has its scores split into two chunks, deliberately at very
different scales (about half the rows have their true max in chunk 1, half
in chunk 2) so a merge that doesn't rescale correctly is caught. The
kernel receives only each chunk's own local (max, sum-of-exp) -- never the
raw scores -- and must merge them into the statistics for the row's full
sequence. The oracle computes those same statistics directly from the
concatenated raw scores, independently of the merge formula.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 64, 64  # one block, one thread per row, no tail
B1, B2 = 8, 12


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(17)
    # Half the rows have chunk 2 dominate in scale, half chunk 1 -- forces
    # both "the new global max came from block 1" and "...block 2".
    offsets = np.where(rng.uniform(size=N) < 0.5, 40.0, -40.0)
    chunk1 = rng.uniform(-5.0, 5.0, size=(N, B1))
    chunk2 = rng.uniform(-5.0, 5.0, size=(N, B2)) + offsets[:, None]

    m1 = chunk1.max(axis=1)
    l1 = np.exp(chunk1 - m1[:, None]).sum(axis=1)
    m2 = chunk2.max(axis=1)
    l2 = np.exp(chunk2 - m2[:, None]).sum(axis=1)

    full = np.concatenate([chunk1, chunk2], axis=1)
    m_ref = full.max(axis=1)
    l_ref = np.exp(full - m_ref[:, None]).sum(axis=1)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(6 * N)
    gpu.gmem[0 * N:1 * N] = 0.0   # m_out
    gpu.gmem[1 * N:2 * N] = 0.0   # l_out
    gpu.gmem[2 * N:3 * N] = m1
    gpu.gmem[3 * N:4 * N] = l1
    gpu.gmem[4 * N:5 * N] = m2
    gpu.gmem[5 * N:6 * N] = l2

    params = {"m_out": 0 * N, "l_out": 1 * N, "m1": 2 * N, "l1": 3 * N,
              "m2": 4 * N, "l2": 5 * N, "n": N}
    try:
        prog.launch(gpu, "merge_online_softmax", N // BLOCK, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    m_got = gpu.gmem[0 * N:1 * N]
    l_got = gpu.gmem[1 * N:2 * N]
    err = max(float(np.max(np.abs(m_got - m_ref))), float(np.max(np.abs(l_got - l_ref))))
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
