"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU), once per (T, load_cycles,
compute_cycles) scenario, comparing against an independently computed oracle.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [(10, 50, 30), (20, 40, 60), (8, 100, 20), (5, 20, 20), (15, 70, 10)]


def _oracle(T, load_cycles, compute_cycles):
    single_total = T * (load_cycles + compute_cycles)
    mx = max(load_cycles, compute_cycles)
    double_total = load_cycles + (T - 1) * mx + compute_cycles
    return float(single_total), float(double_total)


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"exact_match": 0.0, "error": str(e)}

    for T, load_cycles, compute_cycles in CASES:
        gpu = GPU(2)
        gpu.gmem[0:2] = 0.0
        params = {"T": T, "load_cycles": load_cycles, "compute_cycles": compute_cycles, "out": 0}
        try:
            prog.launch(gpu, "buffering_cycles", 1, 1, params)
        except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
            return {"exact_match": 0.0, "error": str(e)}
        single_ref, double_ref = _oracle(T, load_cycles, compute_cycles)
        if float(gpu.gmem[0]) != single_ref or float(gpu.gmem[1]) != double_ref:
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
