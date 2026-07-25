"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram,
statically walk its own AST to model per-thread register pressure, and run
it thread-by-thread on the software GPU (arena.cuda_sim.GPU) for
correctness.

This simulator has no real register allocator (nothing in `arena.cuda_sim`
tracks registers), so "spill" here is a MODELED proxy computed the honest
way available: count the DISTINCT scalar local variables (loop counters
included, __shared__ declarations excluded -- those aren't registers) the
kernel's own parsed source declares. A thread that coarsens N elements by
declaring N separate named temporaries needs all of them alive across most
of the function; a thread that coarsens the same N elements through a loop
with one reused temporary needs only a handful of names regardless of N.
`spill = 1` once that count exceeds a fixed per-thread register budget.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram, VarDecl, SharedDecl, Block, If, For, While  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK, GRID, COARSEN = 32, 4, 8
N = BLOCK * GRID * COARSEN
C = 1.5
REGISTER_BUDGET = 10


def _count_local_vars(fn) -> int:
    """Distinct scalar local variable names declared anywhere in the
    kernel body (function parameters and __shared__ arrays don't count --
    parameters are register-resident by the ABI regardless of coarsening,
    and __shared__ storage is not a register at all)."""
    names = set()

    def walk(stmt):
        if stmt is None:
            return
        t = type(stmt)
        if t is VarDecl:
            names.add(stmt.name)
        elif t is SharedDecl:
            pass  # shared memory, not a register
        elif t is Block:
            for s in stmt.stmts:
                walk(s)
        elif t is If:
            walk(stmt.then)
            walk(stmt.els)
        elif t is For:
            walk(stmt.init)
            walk(stmt.body)
        elif t is While:
            walk(stmt.body)
        # ExprStmt/Return/Sync declare nothing

    walk(fn.body)
    return len(names)


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "spill": 1, "modeled_registers": 10 ** 9, "error": str(e)}

    fn = prog.funcs.get("coarsened_square")
    if fn is None:
        return {"max_abs_err": float("inf"), "spill": 1, "modeled_registers": 10 ** 9,
                "error": "no such kernel 'coarsened_square'"}

    modeled_registers = _count_local_vars(fn)
    spill = 1 if modeled_registers > REGISTER_BUDGET else 0

    rng = np.random.RandomState(5)
    x = rng.uniform(-3.0, 3.0, size=N)
    ref = x * x + C

    gpu = GPU(2 * N)
    gpu.gmem[0:N] = x
    gpu.gmem[N:2 * N] = 0.0
    params = {"out": N, "in": 0, "n": N, "c": C}
    try:
        prog.launch(gpu, "coarsened_square", GRID, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "spill": spill, "modeled_registers": modeled_registers,
                "error": str(e)}

    got = gpu.gmem[N:2 * N]
    max_err = float(np.max(np.abs(got - ref)))
    return {"max_abs_err": max_err, "spill": spill, "modeled_registers": modeled_registers}


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
