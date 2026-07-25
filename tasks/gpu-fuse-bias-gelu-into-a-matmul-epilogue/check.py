"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU). Compares
a fused matmul-bias-GELU epilogue against a numpy oracle and reports
`transactions`/`cycles` -- a kernel that applies bias+GELU to its own
accumulator register before ever writing to global memory touches memory far
less than one that writes the raw matmul result out, syncs, reads it back,
and writes the activated result over it.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M, K, N = 4, 8, 4
NA, NB, NBIAS, NOUT = M * K, K * N, N, M * N


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(21)
    A = rng.randn(NA).astype(np.float64)
    B = rng.randn(NB).astype(np.float64)
    bias = (rng.randn(NBIAS) * 0.3).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"rel_err": float("inf"), "transactions": 10 ** 9, "cycles": 10 ** 9, "error": str(e)}

    gpu = GPU(NA + NB + NBIAS + NOUT, smem_size=1)
    gpu.gmem[0:NA] = A                                  # A    = gmem[0:NA],           shape (M,K)
    gpu.gmem[NA:NA + NB] = B                            # B    = gmem[NA:NA+NB],       shape (K,N)
    gpu.gmem[NA + NB:NA + NB + NBIAS] = bias            # bias = gmem[NA+NB:..],       shape (N,)
    gpu.gmem[NA + NB + NBIAS:NA + NB + NBIAS + NOUT] = 0.0  # out = gmem[..:..+NOUT]

    params = {"out": NA + NB + NBIAS, "A": 0, "B": NA, "bias": NA + NB, "M": M, "K": K, "N": N}
    try:
        m = prog.launch(gpu, "matmul_bias_gelu", 1, M * N, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"rel_err": float("inf"), "transactions": 10 ** 9, "cycles": 10 ** 9, "error": str(e)}

    z = A.reshape(M, K) @ B.reshape(K, N) + bias.reshape(1, N)
    # tanh-approximation GELU (the same closed form the kernel must use,
    # expressed here via the exact tanh identity so there is no separate
    # approximation error between the oracle and a correct kernel).
    ref_out = 0.5 * z * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))

    got = gpu.gmem[NA + NB + NBIAS:NA + NB + NBIAS + NOUT].reshape(M, N)
    denom = max(float(np.max(np.abs(ref_out))), 1e-12)
    rel_err = float(np.max(np.abs(got - ref_out)) / denom)
    return {"rel_err": rel_err, "transactions": int(m["transactions"]), "cycles": int(m["cycles"])}


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
