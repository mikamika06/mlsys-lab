"""Module for tracing PyTorch models and exporting them to Core ML."""


def export_and_verify(model, example_inputs, eval_inputs, save_path):
    """
    Traces a PyTorch model with example_inputs, exports to save_path as .mlpackage,
    runs both models on eval_inputs, and returns (mlmodel, max_abs_err).
    """
    raise NotImplementedError
