"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU), once per (sram_bytes,
head_dim) scenario, comparing against an independently computed oracle
(never hardcoded -- derived the same way, in Python, from first principles).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [(49152, 64), (98304, 64), (49152, 128), (16384, 32), (65536, 16)]


def _oracle(sram_bytes: int, head_dim: int) -> float:
    # Q, K, V, O tiles: Q and O are Br x head_dim, K and V are Bc x head_dim.
    # Square-tiling (Br = Bc = T): total bytes = 4 * T * head_dim * 4 (float).
    raw = sram_bytes // (16 * head_dim)
    if raw < 1:
        return 0.0
    e = math.floor(math.log2(raw) + 1e-9)
    return float(2 ** e)


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"exact_match": 0.0, "error": str(e)}

    for sram_bytes, head_dim in CASES:
        gpu = GPU(1)
        gpu.gmem[0] = 0.0
        params = {"sram_bytes": sram_bytes, "head_dim": head_dim, "out": 0}
        try:
            prog.launch(gpu, "derive_tile_size", 1, 1, params)
        except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
            return {"exact_match": 0.0, "error": str(e)}
        if float(gpu.gmem[0]) != _oracle(sram_bytes, head_dim):
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
