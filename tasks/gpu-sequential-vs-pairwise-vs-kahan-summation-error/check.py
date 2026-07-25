"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it (single thread) on the software GPU (arena.cuda_sim.GPU). The
fixture is deliberately ill-conditioned (a huge value, many small values,
then the huge value negated back off) -- extreme enough that even this
simulator's double-precision backend shows the same catastrophic-
cancellation failure a naive fp32 sum shows on real hardware, and Kahan
summation fixes it exactly the same way.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_ONES = 1000
VALUES = [1e16] + [1.0] * N_ONES + [-1e16]
TRUE_SUM = float(N_ONES)


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"rel_err": float("inf"), "error": str(e)}

    n = len(VALUES)
    gpu = GPU(1 + n)
    gpu.gmem[0] = -1.0
    gpu.gmem[1:1 + n] = VALUES

    try:
        prog.launch(gpu, "kahan_sum", 1, 1, {"out": 0, "values": 1, "n": n})
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"rel_err": float("inf"), "error": str(e)}

    got = float(gpu.gmem[0])
    rel_err = abs(got - TRUE_SUM) / abs(TRUE_SUM)
    return {"rel_err": rel_err}


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
