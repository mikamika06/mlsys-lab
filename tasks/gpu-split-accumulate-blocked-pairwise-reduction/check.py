"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it (single thread) on the software GPU (arena.cuda_sim.GPU). The
fixture places a huge-magnitude cancelling pair (1e20, -1e20) in the MIDDLE
of a run of small values -- large enough that even this simulator's
double-precision backend completely swallows any small value that
combines with it directly, but a block boundary that isolates the pair
into its own group protects everything else.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HUGE = 1e20
VALUES = [1.0] * 30 + [HUGE, -HUGE] + [1.0] * 32
BLOCK_SIZE = 2
TRUE_SUM = 62.0


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
        prog.launch(gpu, "split_accumulate", 1, 1,
                    {"out": 0, "values": 1, "n": n, "block_size": BLOCK_SIZE})
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
