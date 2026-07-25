"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it thread-by-thread on the software GPU (arena.cuda_sim.GPU).

The lesson: a reduction whose per-block combining order depends on runtime
scheduling (like a real atomicAdd-based reduction, where concurrent threads'
updates land in whatever order they finish) is only bitwise-reproducible if
floating-point addition is associative -- and it isn't. This grader feeds the
SAME 64 values, permuted M different ways, through the candidate's
block-level tree reduction, and checks two things: every individual sum
matches what the reference implementation gets for that same permutation
(correctness), and the resulting spread between the highest and lowest sum
across all M permutations never exceeds a bound derived from the classical
Wilkinson summation-error formula (the actual, provable worst case).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 64          # values per block, must equal blockDim
M = 40          # number of independent orderings tried
DBL_EPS = 2.0 ** -52  # machine epsilon for the simulator's float64 arithmetic


def _build_values():
    rng = np.random.RandomState(3)
    big = 1.0e16 + rng.uniform(-1e10, 1e10, size=N // 2)
    small = rng.uniform(-5.0, 5.0, size=N // 2)
    return np.concatenate([big, small])


def _run_all(prog, x, perms):
    sums = []
    for perm in perms:
        xv = x[perm]
        gpu = GPU(N + 1, smem_size=N)
        gpu.gmem[0:N] = xv
        gpu.gmem[N] = 0.0
        try:
            prog.launch(gpu, "block_reduce_sum", 1, N, {"x": 0, "out": N, "n": N})
        except Exception:  # noqa: BLE001 -- any runtime fault fails the gate cleanly
            sums.append(float("nan"))
            continue
        sums.append(float(gpu.gmem[N]))
    return sums


def grade(srcfile: str = "solve.cu") -> dict:
    x = _build_values()
    perms = [np.random.RandomState(1000 + t).permutation(N) for t in range(M)]

    with open(os.path.join(HERE, "ref.cu")) as f:
        ref_prog = CudaProgram(f.read())
    ref_sums = _run_all(ref_prog, x, perms)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()
    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"sums_match": 0.0, "within_bound": 0.0, "error": str(e)}
    cand_sums = _run_all(prog, x, perms)

    sums_match = 1.0 if all(
        c == r for c, r in zip(cand_sums, ref_sums)
    ) else 0.0

    ref_spread = max(ref_sums) - min(ref_sums)
    bound = 2.0 * (N - 1) * DBL_EPS * float(np.sum(np.abs(x)))
    within_bound = 1.0 if ref_spread <= bound else 0.0

    return {
        "sums_match": sums_match,
        "within_bound": within_bound,
        "spread": ref_spread,
        "bound": bound,
    }


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
