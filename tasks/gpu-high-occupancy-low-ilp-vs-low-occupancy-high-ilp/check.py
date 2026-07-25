"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU), once per (warps_resident,
ilp, compute_cycles, mem_latency) scenario, comparing against an
independently computed oracle.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# (warps_resident, ilp, compute_cycles, mem_latency) -- the first two rows
# are the high-occupancy/low-ILP vs low-occupancy/high-ILP regimes.
CASES = [(32, 1, 200, 256), (8, 32, 200, 256), (16, 4, 150, 300), (4, 64, 150, 300), (32, 8, 100, 256)]


def _oracle(w, ilp, compute_cycles, mem_latency):
    concurrency = w * ilp
    exposed = max(0, mem_latency - concurrency)
    return float(compute_cycles + exposed)


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"exact_match": 0.0, "error": str(e)}

    for w, ilp, compute_cycles, mem_latency in CASES:
        gpu = GPU(1)
        gpu.gmem[0] = 0.0
        params = {"warps_resident": w, "ilp": ilp, "compute_cycles": compute_cycles,
                  "mem_latency": mem_latency, "out": 0}
        try:
            prog.launch(gpu, "latency_hiding_cycles", 1, 1, params)
        except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
            return {"exact_match": 0.0, "error": str(e)}
        if float(gpu.gmem[0]) != _oracle(w, ilp, compute_cycles, mem_latency):
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
