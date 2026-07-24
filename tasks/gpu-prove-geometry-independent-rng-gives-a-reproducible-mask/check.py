"""Grade a REAL CUDA-C dropout-mask kernel: parse solve.cu with
arena.cuda_c.CudaProgram and execute it thread-by-thread on the software
GPU (arena.cuda_sim.GPU). Checks the mask matches a CPU-computed reference
AND that two different launch geometries covering the same N elements
produce the IDENTICAL mask (proving the per-element RNG depends only on the
element index, never on thread/block shape).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 1024
SEED = 12345
LIMIT = 128
MOD = 4294967296    # 2**32, emulated with an explicit % (no bitwise ops in this CUDA-C subset)
MULT = 2654435761   # Knuth multiplicative hash constant


def _ref_mask():
    mask = []
    for i in range(N):
        h = (SEED + i * MULT) % MOD
        r = (h // 16777216) % 256
        mask.append(1.0 if r < LIMIT else 0.0)
    return mask


def _run(prog, grid, block):
    gpu = GPU(N)
    params = {"mask": 0, "seed": SEED, "n": N, "limit": LIMIT}
    prog.launch(gpu, "dropout_mask_kernel", grid, block, params)
    return list(gpu.gmem[:N])


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"ref_match": 0.0, "geom_match": 0.0, "error": str(e)}

    ref = _ref_mask()

    block1, grid1 = 64, (N + 63) // 64
    block2, grid2 = 32, (N + 31) // 32

    try:
        mask1 = _run(prog, grid1, block1)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"ref_match": 0.0, "geom_match": 0.0, "error": str(e)}

    ref_match = sum(1 for a, b in zip(mask1, ref) if a == b) / N

    try:
        mask2 = _run(prog, grid2, block2)
    except Exception as e:  # noqa: BLE001
        return {"ref_match": ref_match, "geom_match": 0.0, "error": str(e)}

    geom_match = sum(1 for a, b in zip(mask1, mask2) if a == b) / N

    return {"ref_match": ref_match, "geom_match": geom_match}


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
