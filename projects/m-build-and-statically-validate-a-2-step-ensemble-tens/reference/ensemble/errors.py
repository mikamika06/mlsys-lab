def classify_ensemble_error(err_str):
    if "input" in err_str and "not produced" in err_str:
        return "MISSING_TENSOR_INPUT"
    if "data type mismatch" in err_str:
        return "TYPE_MISMATCH"
    if "cyclic dependency" in err_str:
        return "CYCLIC_DEPENDENCY"
    return "UNKNOWN"
