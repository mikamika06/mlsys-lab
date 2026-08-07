import warnings
import ref

def check(workdir):
    from compressor_workflow.quantize import execute_oneshot
    out = {"quantization_success": 0.0, "deprecation_captured": 0.0}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = execute_oneshot("small-model-stub")
            if res is not None:
                out["quantization_success"] = 1.0
        except Exception as e:
            out["_note"] = f"execute_oneshot failed: {e}"
            return out

        deprecation_found = any(issubclass(warn.category, DeprecationWarning) and "2:4 sparsity" in str(warn.message) for warn in w)
        if deprecation_found:
            out["deprecation_captured"] = 1.0
        else:
            out["_note"] = "Deprecation warning for 2:4 sparsity not captured during execution."

    return out
