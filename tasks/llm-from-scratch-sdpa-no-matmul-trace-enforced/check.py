import ast
import sys
import inspect
from math import exp, sqrt

import numpy as np

from mlsys import scorers

# numpy attributes trapped at runtime: calling any of them raises, so a solver
# that reaches for a matmul primitive fails instead of quietly vectorizing.
_BAN = ("matmul", "dot", "einsum", "tensordot", "inner", "vdot")


def _oracle(Q, K, V):
    """Trusted vectorized NumPy SDPA (this is the check's own reference)."""
    dk = Q.shape[-1]
    s = (Q @ K.T) / np.sqrt(dk)
    e = np.exp(s - np.max(s, axis=-1, keepdims=True))
    p = e / np.sum(e, axis=-1, keepdims=True)
    return p @ V


def _ref_loops(Q, K, V):
    """A genuine from-scratch triple-loop SDPA.

    Only used to MEASURE the expected number of Python line-events for an honest
    O(S^2 d) loop implementation on a given input -- the op-count band is derived
    from this live measurement, never hardcoded.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    S, d = Q.shape
    Sk = K.shape[0]
    dv = V.shape[1]
    scale = 1.0 / sqrt(d)

    scores = np.empty((S, Sk), dtype=np.float64)
    for i in range(S):
        for j in range(Sk):
            acc = 0.0
            for t in range(d):
                acc += Q[i, t] * K[j, t]
            scores[i, j] = acc * scale

    out = np.empty((S, dv), dtype=np.float64)
    for i in range(S):
        m = scores[i, 0]
        for j in range(1, Sk):
            if scores[i, j] > m:
                m = scores[i, j]
        denom = 0.0
        w = np.empty(Sk, dtype=np.float64)
        for j in range(Sk):
            w[j] = exp(scores[i, j] - m)
            denom += w[j]
        for j in range(Sk):
            w[j] /= denom
        for t in range(dv):
            acc = 0.0
            for j in range(Sk):
                acc += w[j] * V[j, t]
            out[i, t] = acc
    return out


def _banned(*args, **kwargs):
    raise RuntimeError(
        "matmul/dot/einsum/tensordot/inner are banned -- do the contraction "
        "with explicit loops"
    )


def _run(fn, *args):
    """Call fn(*args) under sys.settrace, returning (output, line_event_count).

    Only line-events executed inside ``fn``'s OWN code object are counted -- so
    numpy's C kernels AND its Python dispatch wrappers (``np.sum`` etc.) are
    excluded, and what remains is exactly the explicit loop body the solver
    wrote. A vectorized contraction runs in C/numpy and emits (almost) none,
    landing far below the band; an honest O(S^2 d) loop lands near 1.0x. This
    mirrors the counting convention already used elsewhere in the arena.
    """
    count = 0
    code = getattr(fn, "__code__", None)

    def tracer(frame, event, arg):
        nonlocal count
        if event == "line" and frame.f_code is code:
            count += 1
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        out = fn(*args)
    finally:
        sys.settrace(old)
    return out, count


_BANNED_ATTRS = {"dot", "matmul", "einsum", "tensordot", "inner", "vdot"}


def _uses_banned_source(sol):
    """True if the solver's source uses @ or a matmul/dot/einsum primitive.

    Parsed via the AST so string literals, docstrings and comments are ignored,
    and a decorator (which lives in ``decorator_list``, not a ``MatMult`` node)
    is never mistaken for matrix multiplication. Catches the operator forms the
    runtime trap cannot: ``@`` maps to the C-level ``ndarray.__matmul__`` and
    ``x.dot(...)`` is a C method, both bypassing the patched numpy functions.
    """
    try:
        src = inspect.getsource(sol)
    except (OSError, TypeError):
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult):
            return True
        if isinstance(node, ast.Attribute) and node.attr in _BANNED_ATTRS:
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _BANNED_ATTRS:
            return True
    return False


def grade(sol, fx) -> dict:
    if _uses_banned_source(sol):
        return {"max_abs_err": float("inf"), "op_ratio": 0.0}

    rng = np.random.default_rng(0)
    cases = [(8, 6), (5, 4)]  # op-count band is measured on the first (larger) case

    max_err = 0.0
    op_ratio = 0.0

    for idx, (S, d) in enumerate(cases):
        Q = rng.standard_normal((S, d))
        K = rng.standard_normal((S, d))
        V = rng.standard_normal((S, d))
        ref = _oracle(Q, K, V)

        saved = {name: getattr(np, name) for name in _BAN}
        for name in _BAN:
            setattr(np, name, _banned)
        try:
            got, cand_count = _run(sol.sdpa, Q, K, V)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            got, cand_count = None, 0
        finally:
            for name, value in saved.items():
                setattr(np, name, value)

        if got is None or got.shape != ref.shape:
            return {"max_abs_err": float("inf"), "op_ratio": 0.0}

        err = scorers.max_abs_err(ref, got)
        if err > max_err:
            max_err = err

        if idx == 0:
            _, exp_count = _run(_ref_loops, Q, K, V)
            op_ratio = float(cand_count) / exp_count if exp_count > 0 else 0.0

    return {"max_abs_err": max_err, "op_ratio": op_ratio}
