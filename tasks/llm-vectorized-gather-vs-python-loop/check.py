import sys
import numpy as np
from mlsys.scorers import max_abs_err

def _reference(indices, embedding_matrix):
    return np.asarray(embedding_matrix, dtype=np.float64)[indices]

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(seed=42)
    max_err = 0.0
    max_op = 0

    def trace_lines(frame, event, arg):
        nonlocal op_counter
        if event == 'line':
            op_counter += 1
        return trace_lines

    def trace_calls(frame, event, arg):
        nonlocal op_counter
        if event == 'call':
            code_obj = frame.f_code
            if code_obj.co_name == sol.gather_embeddings.__name__:
                return trace_lines
        return None

    for _ in range(5):
        vocab = rng.integers(1000, 2000)
        dim = rng.integers(64, 256)
        embed = rng.standard_normal((vocab, dim)).astype(np.float64)
        n = rng.integers(50, 300)
        indices = rng.integers(0, vocab, size=n)

        ref = _reference(indices, embed)

        op_counter = 0
        sys.settrace(trace_calls)
        try:
            out = sol.gather_embeddings(indices, embed)
        except Exception:
            return {"max_abs_err": float("inf"), "op_count": max_op}
        finally:
            sys.settrace(None)

        if not isinstance(out, np.ndarray):
            return {"max_abs_err": float("inf"), "op_count": max_op}

        err = max_abs_err(ref, out)
        if err > max_err:
            max_err = err
        if op_counter > max_op:
            max_op = op_counter

    return {"max_abs_err": max_err, "op_count": max_op}
