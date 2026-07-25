"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it (single thread) on the software GPU (arena.cuda_sim.GPU). Compares
the (decode_ai, prefill_ai, decode_bound, prefill_bound) quadruple against a
Python oracle computed with the same fixed formulas.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D_IN, D_OUT, T = 4096.0, 4096.0, 128.0
PEAK_FLOPS, PEAK_BW = 1000.0, 100.0


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(4)
    gpu.gmem[:] = -1.0
    params = {"out": 0, "d_in": D_IN, "d_out": D_OUT, "t": T,
              "peak_flops": PEAK_FLOPS, "peak_bw": PEAK_BW}
    try:
        prog.launch(gpu, "decode_prefill_ai", 1, 1, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ridge = PEAK_FLOPS / PEAK_BW
    decode_ai = (2 * D_IN * D_OUT) / (4 * (D_IN * D_OUT + D_IN + D_OUT))
    prefill_ai = (2 * T * D_IN * D_OUT) / (4 * (D_IN * D_OUT + T * D_IN + T * D_OUT))
    ref = [decode_ai, prefill_ai,
           1.0 if decode_ai >= ridge else 0.0,
           1.0 if prefill_ai >= ridge else 0.0]

    got = gpu.gmem
    max_err = float(max(abs(got[i] - ref[i]) for i in range(4)))
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
