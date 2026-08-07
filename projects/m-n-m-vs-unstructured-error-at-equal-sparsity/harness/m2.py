import ref


def check(workdir):
    from edge_export.joint_budget import find_optimal_joint_budget

    out = {"joint_budget_correct": 0.0, "optimal_combination_selected": 0.0}

    weights = ref.LAYER_WEIGHTS_1[0]
    max_bits = 2.0
    bit_options = [2, 4, 8]

    res = find_optimal_joint_budget(weights, max_bits, bit_options)
    ref_res = ref.check_joint_budget(weights, max_bits, bit_options)

    if (
        isinstance(res, dict)
        and "effective_bits" in res
        and "mse" in res
        and "use_nm" in res
    ):
        if res["effective_bits"] <= max_bits + 1e-6:
            out["joint_budget_correct"] = 1.0

        if (
            res["n"] == ref_res["n"]
            and res["m"] == ref_res["m"]
            and res["bits"] == ref_res["bits"]
            and res["use_nm"] == ref_res["use_nm"]
            and abs(res["mse"] - ref_res["mse"]) < 1e-5
        ):
            out["optimal_combination_selected"] = 1.0
        else:
            out["_note"] = f"Got {res}, expected {ref_res}"

    return out
