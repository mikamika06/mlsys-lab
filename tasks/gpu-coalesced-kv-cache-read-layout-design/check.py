"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU). Gathers a decode-step
slice out of a [layers, heads, seq_len, dim] KV-cache and checks both the
output VALUES (against a numpy oracle) and the simulator's `transactions`
count (128-byte global-memory segments touched -- the coalescing measure).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS, HEADS, SEQ_LEN, DIM = 4, 8, 16, 32
T = 5
TOTAL = LAYERS * HEADS * DIM


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(3)
    kv = rng.randn(LAYERS * HEADS * SEQ_LEN * DIM).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    gpu = GPU(len(kv) + TOTAL)
    gpu.gmem[0:len(kv)] = kv           # kv  = gmem[0 : LAYERS*HEADS*SEQ_LEN*DIM]
    gpu.gmem[len(kv):len(kv) + TOTAL] = 0.0  # out = gmem[len(kv) : len(kv)+TOTAL]

    params = {"out": len(kv), "kv": 0, "layers": LAYERS, "heads": HEADS,
              "seq_len": SEQ_LEN, "dim": DIM, "t": T}
    try:
        m = prog.launch(gpu, "decode_read", TOTAL // 32, 32, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "transactions": 10 ** 9, "error": str(e)}

    kv4 = kv.reshape(LAYERS, HEADS, SEQ_LEN, DIM)
    oracle = kv4[:, :, T, :].reshape(TOTAL)  # K[:, :, T, :], flattened (layer,head,d)
    max_err = float(np.max(np.abs(gpu.gmem[len(kv):len(kv) + TOTAL] - oracle)))
    return {"max_abs_err": max_err, "transactions": int(m["transactions"])}


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
