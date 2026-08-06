def validate_imatrix(imatrix_obj, model_config):
    """Validate imatrix structure and non-negativity against model config."""
    raise NotImplementedError


def run_and_validate_pipeline(model_config, activation_batches):
    """Run imatrix compute, merge across batches, and validate output."""
    raise NotImplementedError
