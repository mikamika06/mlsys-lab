"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU), once per (M, N, K, C)
scenario, comparing against an independently computed oracle.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [(16, 64, 32, 4), (8, 128, 16, 8), (32, 32, 8, 2), (4, 256, 64, 16), (10, 60, 20, 5)]


def _oracle(M, N, K, C):
    threads = M * (N // C)
    a_loads = threads * K   # A's row loaded once per k, shared across C outputs per thread
    b_loads = M * N * K     # B needs a fresh load per output element per k, always
    return float(a_loads), float(b_loads)


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"exact_match": 0.0, "error": str(e)}

    for M, N, K, C in CASES:
        gpu = GPU(2)
        gpu.gmem[0:2] = 0.0
        params = {"M": M, "N": N, "K": K, "C": C, "out": 0}
        try:
            prog.launch(gpu, "derive_loads", 1, 1, params)
        except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
            return {"exact_match": 0.0, "error": str(e)}
        a_ref, b_ref = _oracle(M, N, K, C)
        if float(gpu.gmem[0]) != a_ref or float(gpu.gmem[1]) != b_ref:
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
