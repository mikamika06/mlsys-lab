import inspect
import numpy as np

def _source_line_count(func):
    src = inspect.getsource(func)
    lines = [line for line in src.splitlines() if line.strip()]
    return len(lines)

def grade(sol, fx) -> dict:
    # Generate random test cases
    rng = np.random.default_rng(42)
    sizes = [(10, 8, 12), (25, 30, 20), (50, 60, 55)]
    ok = 1.0
    for m, k, n in sizes:
        A = rng.standard_normal((m, k))
        B = rng.standard_normal((k, n))
        try:
            C_candidate = sol.matmul_loops(A, B)
        except Exception:
            return {"rel_err": float("inf"), "line_count": 0}
        C_ref = A @ B
        # Relative error
        rel_err = np.linalg.norm(C_candidate - C_ref) / (np.linalg.norm(C_ref) + 1e-12)
        if rel_err > 1e-12:
            ok = 0.0
            break
    # Line count check: forbid use of @ or dot, require at least 5 lines
    src = inspect.getsource(sol.matmul_loops)
    banned_patterns = ["np.dot", "@"]
    if any(pat in src for pat in banned_patterns):
        ok = 0.0
    line_count = _source_line_count(sol.matmul_loops)
    return {"rel_err": ok, "line_count": line_count}
