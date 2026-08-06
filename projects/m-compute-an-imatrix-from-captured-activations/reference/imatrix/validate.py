import numpy as np
from imatrix.compute import compute_imatrix
from imatrix.merge import merge_imatrices


def validate_imatrix(imatrix_obj, model_config):
    """Validate imatrix structure and non-negativity against model config."""
    if not isinstance(imatrix_obj, dict) or "data" not in imatrix_obj:
        return False
    if imatrix_obj.get("count", 0) <= 0:
        return False
    data = imatrix_obj["data"]
    expected_layers = model_config.get("layers", {})
    if set(data.keys()) != set(expected_layers.keys()):
        return False
    for k, dim in expected_layers.items():
        arr = data.get(k)
        if not isinstance(arr, np.ndarray):
            return False
        if arr.ndim != 1 or arr.shape[0] != dim:
            return False
        if np.isnan(arr).any() or np.isinf(arr).any():
            return False
        if np.any(arr < 0.0):
            return False
    return True


def run_and_validate_pipeline(model_config, activation_batches):
    """Run imatrix compute, merge across batches, and validate output."""
    shards = []
    for batch in activation_batches:
        c = batch["count"]
        im = compute_imatrix(batch["activations"])
        shards.append({"count": c, "data": im})
    merged = merge_imatrices(shards)
    is_valid = validate_imatrix(merged, model_config)
    return {"valid": is_valid, "imatrix": merged}
