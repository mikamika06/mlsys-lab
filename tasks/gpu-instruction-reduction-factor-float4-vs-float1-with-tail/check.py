"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it (single thread) on the software GPU (arena.cuda_sim.GPU), once
per fixed n, comparing the derived (loads_float1, loads_float4, ratio)
triple against a computed-in-Python oracle.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NS = (17, 100, 256, 4097)


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    max_err = 0.0
    for n in NS:
        gpu = GPU(3)
        gpu.gmem[:] = -1.0
        try:
            prog.launch(gpu, "float4_instr_counts", 1, 1, {"out": 0, "n": n})
        except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
            return {"max_abs_err": float("inf"), "error": str(e)}

        loads1 = n
        loads4 = n // 4 + n % 4
        ratio = round(loads1 / loads4, 6)
        ref = (loads1, loads4, ratio)
        got = gpu.gmem
        err = max(abs(got[i] - ref[i]) for i in range(3))
        max_err = max(max_err, err)

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
