"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram,
run it thread-by-thread on the software GPU for correctness, AND walk its
own AST to count distinct local-variable declarations in the kernel body
-- a static proxy for registers-per-thread -- feeding a fixed occupancy
model (registers-per-thread * threads-per-block against a fixed register
file, capped by a fixed max-blocks-per-SM).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram, VarDecl, Block, If, For, While  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, BLOCK = 256, 128
A, B, C, D = 1.0, 2.0, -1.0, 0.5

REG_FILE_PER_SM = 2048
MAX_BLOCKS_PER_SM_HW = 8
OCCUPANCY_TARGET = 0.75


def _collect_local_var_names(stmt, names):
    """Walk statement-level AST nodes only -- CUDA-C declarations are
    always statements, never nested inside expressions, so this never
    needs to descend into expression trees."""
    if stmt is None:
        return
    if isinstance(stmt, VarDecl):
        names.add(stmt.name)
    elif isinstance(stmt, Block):
        for s in stmt.stmts:
            _collect_local_var_names(s, names)
    elif isinstance(stmt, If):
        _collect_local_var_names(stmt.then, names)
        _collect_local_var_names(stmt.els, names)
    elif isinstance(stmt, For):
        _collect_local_var_names(stmt.init, names)
        _collect_local_var_names(stmt.body, names)
    elif isinstance(stmt, While):
        _collect_local_var_names(stmt.body, names)


def _modeled_occupancy(regs_per_thread: int) -> float:
    if regs_per_thread <= 0:
        regs_per_thread = 1
    max_blocks_by_regs = REG_FILE_PER_SM // (regs_per_thread * BLOCK)
    blocks = min(max_blocks_by_regs, MAX_BLOCKS_PER_SM_HW)
    return blocks / MAX_BLOCKS_PER_SM_HW


def grade(srcfile: str = "solve.cu") -> dict:
    rng = np.random.RandomState(3)
    x = rng.randn(N).astype(np.float64)

    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "occupancy_ok": 0.0, "error": str(e)}

    fn = prog.funcs.get("sum_sq_dev")
    if fn is None:
        return {"max_abs_err": float("inf"), "occupancy_ok": 0.0, "error": "no sum_sq_dev kernel"}
    names = set()
    _collect_local_var_names(fn.body, names)
    regs_per_thread = len(names)
    occupancy = _modeled_occupancy(regs_per_thread)

    gpu = GPU(2 * N)
    gpu.gmem[0:N] = 0.0
    gpu.gmem[N:2 * N] = x
    params = {"out": 0, "x": N, "a": A, "b": B, "c": C, "d": D, "n": N}
    try:
        prog.launch(gpu, "sum_sq_dev", N // BLOCK, BLOCK, params)
    except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
        return {"max_abs_err": float("inf"), "occupancy_ok": 0.0, "error": str(e)}

    ref_out = (x - A) ** 2 + (x - B) ** 2 + (x - C) ** 2 + (x - D) ** 2
    max_err = float(np.max(np.abs(gpu.gmem[0:N] - ref_out)))
    occupancy_ok = 1.0 if occupancy >= OCCUPANCY_TARGET else 0.0
    return {"max_abs_err": max_err, "occupancy_ok": occupancy_ok}


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
