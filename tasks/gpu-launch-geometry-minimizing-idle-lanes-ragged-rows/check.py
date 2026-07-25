"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU). A ragged batch (rows of
different lengths) is processed by a COMPACT flat launch sized to the real
total element count, not padded to R*MAXLEN. Correctness is checked against
a numpy oracle; the simulator's `transactions` count is compared against a
FIXED, always-correct PADDED baseline (embedded here, never the learner's
code) that launches one thread per padded [R, MAXLEN] slot instead.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R, MAXLEN = 8, 16
LENS = [16, 4, 8, 2, 16, 6, 10, 3]
TOTAL = int(sum(LENS))  # 65 -- real element count, no padding
OFFSETS = [0]
for L in LENS:
    OFFSETS.append(OFFSETS[-1] + L)
OFFSETS = OFFSETS[:R]  # offsets[r] = flat start of row r (R entries)

PADDED = R * MAXLEN  # 128 -- what a naive padded launch would use

# HARNESS baseline (fixed, not learner code): pads every row out to
# MAXLEN and launches one thread per padded slot, whether or not it's
# real data.
PADDED_SRC = """
__global__ void ragged_process(float* out, const float* data, const int* lens,
                                const float* row_scale, int R, int MAXLEN) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid / MAXLEN;
    out[tid] = data[tid] + row_scale[row];
}
"""


def _setup_compact(rng):
    data = rng.uniform(-3.0, 3.0, size=TOTAL)
    row_scale = np.arange(R, dtype=np.float64) * 10.0
    gpu = GPU(TOTAL + TOTAL + R + R, smem_size=1)
    gpu.gmem[0:TOTAL] = 0.0
    gpu.gmem[TOTAL:2 * TOTAL] = data
    gpu.gmem[2 * TOTAL:2 * TOTAL + R] = OFFSETS
    gpu.gmem[2 * TOTAL + R:2 * TOTAL + 2 * R] = row_scale
    params = {"out": 0, "data": TOTAL, "offsets": 2 * TOTAL, "row_scale": 2 * TOTAL + R,
              "R": R, "total": TOTAL}
    return data, row_scale, gpu, params


def _setup_padded(rng):
    data = rng.uniform(-3.0, 3.0, size=PADDED)
    row_scale = np.arange(R, dtype=np.float64) * 10.0
    gpu = GPU(PADDED + PADDED + R + R, smem_size=1)
    gpu.gmem[0:PADDED] = 0.0
    gpu.gmem[PADDED:2 * PADDED] = data
    gpu.gmem[2 * PADDED:2 * PADDED + R] = LENS
    gpu.gmem[2 * PADDED + R:2 * PADDED + 2 * R] = row_scale
    params = {"out": 0, "data": PADDED, "lens": 2 * PADDED, "row_scale": 2 * PADDED + R,
              "R": R, "MAXLEN": MAXLEN}
    return gpu, params


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(17)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
        padded_prog = CudaProgram(PADDED_SRC)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "size_ratio": 0.0, "error": str(e)}

    data, row_scale, gpu, params = _setup_compact(rng)
    try:
        m = prog.launch(gpu, "ragged_process", 1, TOTAL, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "size_ratio": 0.0, "error": str(e)}

    oracle = np.zeros(TOTAL)
    for r in range(R):
        oracle[OFFSETS[r]:OFFSETS[r] + LENS[r]] = (
            data[OFFSETS[r]:OFFSETS[r] + LENS[r]] + row_scale[r])
    got = gpu.gmem[0:TOTAL]
    max_err = float(np.max(np.abs(got - oracle)))

    gpu_pad, params_pad = _setup_padded(rng)
    m_pad = padded_prog.launch(gpu_pad, "ragged_process", 1, PADDED, params_pad)

    warp_ratio = float(m_pad["warps"]) / float(m["warps"])
    return {"max_abs_err": max_err, "warp_ratio": warp_ratio,
            "compact_warps": int(m["warps"]), "padded_warps": int(m_pad["warps"]),
            "compact_transactions": int(m["transactions"]),
            "padded_transactions": int(m_pad["transactions"])}


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
