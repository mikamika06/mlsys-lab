"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it (single thread) on the software GPU (arena.cuda_sim.GPU), once
per fixed (launch_cost, per_elem_cost) case, comparing the derived
crossover N against math.ceil(launch_cost / per_elem_cost).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = ((1000.0, 5.0), (250.0, 3.0), (5000.0, 50.0), (1.0, 7.0))


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    max_err = 0.0
    for launch_cost, per_elem_cost in CASES:
        gpu = GPU(1)
        gpu.gmem[:] = -1.0
        try:
            prog.launch(gpu, "crossover_n", 1, 1,
                        {"out": 0, "launch_cost": launch_cost, "per_elem_cost": per_elem_cost})
        except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
            return {"max_abs_err": float("inf"), "error": str(e)}

        ref = math.ceil(launch_cost / per_elem_cost)
        err = abs(gpu.gmem[0] - ref)
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
