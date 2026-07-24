def _oracle_label(op):
    """Return True if op is export‑incompatible according to the specification."""
    if "item" in op or "nonzero" in op or "bool" in op:
        return True
    if op.endswith("_"):
        return True
    if "python" in op:
        return True
    return False

def grade(sol, fx) -> dict:
    test_cases = [
        ["add", "sub_", "item", "nonzero", "python_side_effect", "mul"],
        ["relu", "sigmoid_", "bool_tensor", "mean", "python_func"],
        ["inplace_add_", "concat", "nonzero", "print", "sum_"],
    ]
    ok = 1.0
    for ops in test_cases:
        try:
            got = sol.flag_export_incompatible_ops(ops)
        except Exception:
            return {"exact_match": 0.0}
        expected = [_oracle_label(op) for op in ops]
        if got != expected:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
