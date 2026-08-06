def compute_formula_params(layer_specs, r, use_dora=False):
    """Compute theoretical trainable parameter count for given layer dimensions."""
    raise NotImplementedError


def count_real_trainable_params(model_layers):
    """Count actual trainable parameters in a dictionary of layer parameters."""
    raise NotImplementedError


def audit_param_counts(layer_specs, r, model_layers, use_dora=False):
    """Compare theoretical parameter formula against real trainable parameter count."""
    raise NotImplementedError
