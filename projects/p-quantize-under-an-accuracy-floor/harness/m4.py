import sys


def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from quant.eval import Evaluator
    from quant.sensitivity import (
        compute_layer_sensitivity,
        select_mixed_precision_config,
    )

    out = {"sensitivity_ranks_correct": 0.0, "mixed_precision_config_valid": 0.0}

    model = ref.MockModel()
    dataset = ref.get_dataset()
    calib = ref.get_calib_data()

    try:
        evaluator = Evaluator(dataset)
        sens = compute_layer_sensitivity(model, evaluator, calib)
    except Exception:
        return out

    if "l1_sensitive" in sens:
        most_sensitive = max(sens.items(), key=lambda x: x[1])[0]
        if most_sensitive == "l1_sensitive":
            out["sensitivity_ranks_correct"] = 1.0

    try:
        config = select_mixed_precision_config(sens)
    except Exception:
        return out

    if isinstance(config, dict) and config.get("l1_sensitive") == 8:
        out["mixed_precision_config_valid"] = 1.0

    return out
