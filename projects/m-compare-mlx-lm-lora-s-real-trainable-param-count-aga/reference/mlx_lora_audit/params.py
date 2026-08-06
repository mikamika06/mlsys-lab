import numpy as np


def compute_formula_params(layer_specs, r, use_dora=False):
    """Compute theoretical trainable parameter count for given layer dimensions."""
    total = 0
    for spec in layer_specs:
        in_dim = spec["in_features"]
        out_dim = spec["out_features"]
        lora_params = r * (in_dim + out_dim)
        dora_params = out_dim if use_dora else 0
        total += lora_params + dora_params
    return total


def count_real_trainable_params(model_layers):
    """Count actual trainable parameters in a dictionary of layer parameters."""
    total = 0
    for layer in model_layers.values():
        for name, param in layer.items():
            if param.get("trainable", True):
                total += int(np.prod(param["weight"].shape))
    return total


def audit_param_counts(layer_specs, r, model_layers, use_dora=False):
    """Compare theoretical parameter formula against real trainable parameter count."""
    formula_count = compute_formula_params(layer_specs, r, use_dora=use_dora)
    real_count = count_real_trainable_params(model_layers)
    return {
        "formula_count": formula_count,
        "real_count": real_count,
        "diff": real_count - formula_count,
        "matches": formula_count == real_count
    }
