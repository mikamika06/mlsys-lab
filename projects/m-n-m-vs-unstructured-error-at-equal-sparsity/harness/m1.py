import ref
import numpy as np


def check(workdir):
    from edge_export.sparsity import compare_sparsity_error, apply_nm_pruning, apply_unstructured_pruning

    out = {"evaluations_correct": 0.0, "nm_error_higher_or_equal": 0.0}

    weights = ref.LAYER_WEIGHTS_1[0]
    res = compare_sparsity_error(weights, 2, 4)
    ref_res = ref.check_sparsity_comparison(weights, 2, 4)

    if (
        isinstance(res, dict)
        and "unstructured_mse" in res
        and "nm_mse" in res
        and abs(res["sparsity_ratio"] - 0.5) < 1e-5
    ):
        if (
            abs(res["unstructured_mse"] - ref_res["unstructured_mse"]) < 1e-5
            and abs(res["nm_mse"] - ref_res["nm_mse"]) < 1e-5
        ):
            out["evaluations_correct"] = 1.0

    if res.get("nm_mse", 0.0) >= res.get("unstructured_mse", 0.0) - 1e-7:
        out["nm_error_higher_or_equal"] = 1.0
    else:
        out["_note"] = f"Expected nm_mse ({res.get('nm_mse')}) >= unstructured_mse ({res.get('unstructured_mse')})"

    return out
