"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it (single thread) on the software GPU (arena.cuda_sim.GPU), once
per fixed cost-model case, comparing the derived (C, AI) pair against a
Python oracle.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# (reg_budget, base_regs, regs_per_c, flops_per_elem, bytes_per_elem)
CASES = (
    (40.0, 6.0, 2.0, 2.0, 4.0),
    (64.0, 8.0, 4.0, 2.0, 4.0),
    (32.0, 10.0, 1.0, 4.0, 4.0),
)


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    max_err = 0.0
    for reg_budget, base_regs, regs_per_c, flops_per_elem, bytes_per_elem in CASES:
        gpu = GPU(2)
        gpu.gmem[:] = -1.0
        params = {"out": 0, "reg_budget": reg_budget, "base_regs": base_regs,
                  "regs_per_c": regs_per_c, "flops_per_elem": flops_per_elem,
                  "bytes_per_elem": bytes_per_elem}
        try:
            prog.launch(gpu, "coarsen_c", 1, 1, params)
        except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
            return {"max_abs_err": float("inf"), "error": str(e)}

        c_ref = math.floor((reg_budget - base_regs) / regs_per_c)
        ai_ref = (flops_per_elem * c_ref) / (bytes_per_elem * (1.0 + c_ref))
        err = max(abs(gpu.gmem[0] - c_ref), abs(gpu.gmem[1] - ai_ref))
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
