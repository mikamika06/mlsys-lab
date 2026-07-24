"""Grade a REAL CUDA-C solve.cu holding three kernels: parse it with
arena.cuda_c.CudaProgram and execute each kernel on the software GPU
(arena.cuda_sim.GPU). Compares against numpy/Python oracles computed here.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 64, 32
K = 5
MODEL_CASES = [(200, 10, 5), (500, 5, 20), (100, 100, 1)]


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(7)
    init = rng.randn(N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "relaunch_slower": 0, "model_correct": float("inf"), "error": str(e)}

    fail = {"max_abs_err": float("inf"), "relaunch_slower": 0, "model_correct": float("inf")}

    # --- 1. persistent_kernel correctness + its single-launch cycle cost ---
    gpu_p = GPU(N)
    gpu_p.gmem[:] = init
    try:
        m_persist = prog.launch(gpu_p, "persistent_kernel", (N + BLOCK - 1) // BLOCK, BLOCK,
                                 {"gmem": 0, "N": N, "K": K})
    except Exception as e:  # noqa: BLE001
        fail["error"] = str(e)
        return fail

    ref_out = init + K
    max_err = float(np.max(np.abs(gpu_p.gmem - ref_out)))

    # --- 2. relaunch_kernel run K times; sum cycles, compare vs persistent ---
    gpu_r = GPU(N)
    gpu_r.gmem[:] = init
    total_relaunch_cycles = 0
    try:
        for _ in range(K):
            m = prog.launch(gpu_r, "relaunch_kernel", (N + BLOCK - 1) // BLOCK, BLOCK, {"gmem": 0, "N": N})
            total_relaunch_cycles += m["cycles"]
    except Exception as e:  # noqa: BLE001
        fail["error"] = str(e)
        return fail

    relaunch_slower = 1 if total_relaunch_cycles > m_persist["cycles"] else 0

    # --- 3. model_launch_cycles_kernel vs the plain arithmetic formulas ---
    model_err = 0.0
    for h, c, k in MODEL_CASES:
        gpu_m = GPU(2)
        gpu_m.gmem[:] = -1.0
        try:
            prog.launch(gpu_m, "model_launch_cycles_kernel", 1, 1,
                        {"out": 0, "launch_overhead": h, "compute_cost_per_iter": c, "K": k})
        except Exception as e:  # noqa: BLE001
            fail["error"] = str(e)
            return fail
        ref_persist = h + k * c
        ref_relaunch = k * (h + c)
        model_err = max(model_err, abs(float(gpu_m.gmem[0]) - ref_persist), abs(float(gpu_m.gmem[1]) - ref_relaunch))

    return {"max_abs_err": max_err, "relaunch_slower": relaunch_slower, "model_correct": model_err}


if __name__ == "__main__":
    import json

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
