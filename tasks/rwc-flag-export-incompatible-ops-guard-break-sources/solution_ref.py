def flag_export_incompatible_ops(ops):
    """
    Return a list of booleans indicating whether each operation name is export‑incompatible.
    """
    result = []
    for op in ops:
        if ("item" in op) or ("nonzero" in op) or ("bool" in op):
            result.append(True)
        elif op.endswith("_"):
            result.append(True)
        elif "python" in op:
            result.append(True)
        else:
            result.append(False)
    return result
