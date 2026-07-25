"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it (single thread) on the software GPU (arena.cuda_sim.GPU). Compares
the 4 arithmetic-intensity values and 4 compute/memory-bound labels against a
Python oracle computed with the same fixed formulas.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GEMM_M = GEMM_K = GEMM_N = 64.0
ATTN_S = ATTN_D = 64.0
LN_N = 4096.0
EW_N = 4096.0
PEAK_FLOPS, PEAK_BW = 1000.0, 100.0


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(8)
    gpu.gmem[:] = -1.0
    params = {"out": 0, "gemm_m": GEMM_M, "gemm_k": GEMM_K, "gemm_n": GEMM_N,
              "attn_s": ATTN_S, "attn_d": ATTN_D, "ln_n": LN_N, "ew_n": EW_N,
              "peak_flops": PEAK_FLOPS, "peak_bw": PEAK_BW}
    try:
        prog.launch(gpu, "roofline_rank", 1, 1, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ridge = PEAK_FLOPS / PEAK_BW
    gemm_ai = (2 * GEMM_M * GEMM_K * GEMM_N) / (4 * (GEMM_M * GEMM_K + GEMM_K * GEMM_N + GEMM_M * GEMM_N))
    attn_ai = (4 * ATTN_S * ATTN_S * ATTN_D) / (4 * (3 * ATTN_S * ATTN_D + ATTN_S * ATTN_S + ATTN_S * ATTN_D))
    ln_ai = (5 * LN_N) / (4 * 2 * LN_N)
    ew_ai = EW_N / (4 * 2 * EW_N)
    ais = (gemm_ai, attn_ai, ln_ai, ew_ai)
    bounds = tuple(1.0 if ai >= ridge else 0.0 for ai in ais)
    ref = list(ais) + list(bounds)

    got = gpu.gmem
    max_err = float(max(abs(got[i] - ref[i]) for i in range(8)))
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
