"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU), once per scenario,
comparing against an independently computed oracle.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [(10, 6, 12), (10, 6, 32), (10, 6, 0), (4, 20, 16), (8, 8, 1), (15, 3, 31)]


def _oracle(then_i, else_i, lanes_then):
    if lanes_then == 0:
        issues = else_i
    elif lanes_then == 32:
        issues = then_i
    else:
        issues = then_i + else_i
    baseline = max(then_i, else_i)
    return float(issues), issues / baseline


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"exact_match": 0.0, "error": str(e)}

    for then_i, else_i, lanes_then in CASES:
        gpu = GPU(2)
        gpu.gmem[0:2] = 0.0
        params = {"then_instrs": then_i, "else_instrs": else_i,
                  "lanes_taking_then": lanes_then, "out": 0}
        try:
            prog.launch(gpu, "divergent_issue_count", 1, 1, params)
        except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
            return {"exact_match": 0.0, "error": str(e)}
        issues_ref, penalty_ref = _oracle(then_i, else_i, lanes_then)
        if float(gpu.gmem[0]) != issues_ref or abs(float(gpu.gmem[1]) - penalty_ref) > 1e-9:
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}


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
