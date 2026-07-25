"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
both the packed slots and the round-tripped codes against a numpy oracle --
byte-exact packing means every value survives pack-then-unpack unchanged.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 64, 32  # 64 int4 codes -> 32 packed slots

RT_BASE = 0
PACKED_BASE = N
CODES_BASE = PACKED_BASE + N // 2
GMEM_SIZE = CODES_BASE + N


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(11)
    codes = rng.randint(0, 16, size=N).astype(np.float64)  # each code in [0,16)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    gpu = GPU(GMEM_SIZE)
    gpu.gmem[RT_BASE:RT_BASE + N] = -1.0
    gpu.gmem[PACKED_BASE:PACKED_BASE + N // 2] = -1.0
    gpu.gmem[CODES_BASE:CODES_BASE + N] = codes

    params = {"roundtrip": RT_BASE, "packed": PACKED_BASE, "codes": CODES_BASE, "n": N}
    try:
        prog.launch(gpu, "pack_unpack_int4", N // 2 // BLOCK + (1 if (N // 2) % BLOCK else 0), BLOCK, params)
    except Exception as e:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "error": str(e)}

    ref_packed = codes[0::2] + 16.0 * codes[1::2]
    packed_err = float(np.max(np.abs(gpu.gmem[PACKED_BASE:PACKED_BASE + N // 2] - ref_packed)))
    roundtrip_err = float(np.max(np.abs(gpu.gmem[RT_BASE:RT_BASE + N] - codes)))
    max_err = max(packed_err, roundtrip_err)
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
