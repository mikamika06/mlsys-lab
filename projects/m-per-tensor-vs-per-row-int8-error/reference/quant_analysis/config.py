import numpy as np
from quant_analysis.error import compare_error_metrics


def pick_torchao_config(w: np.ndarray, max_mse: float) -> str:
    errs = compare_error_metrics(w)
    if errs["per_tensor_mse"] <= max_mse:
        return "int8_weight_only_per_tensor"
    return "int8_weight_only_per_row"


def select_model_configs(weights: dict[str, np.ndarray], max_mse: float) -> dict[str, str]:
    return {layer_name: pick_torchao_config(w, max_mse) for layer_name, w in weights.items()}
