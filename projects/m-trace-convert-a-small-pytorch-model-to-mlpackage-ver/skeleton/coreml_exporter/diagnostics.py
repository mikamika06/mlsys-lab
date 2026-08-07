"""Module for handling unsupported ops during Core ML conversion."""


def safe_convert_or_diagnose(model, example_inputs, unsupported_op_names):
    """
    Attempts conversion. If unsupported ops occur or are present in graph,
    catches error and returns status dict with success, error_msg, unsupported_found.
    """
    raise NotImplementedError
