def import_and_compile(model, example_inputs, supported_ops):
    """Import PyTorch model to Relax representation and return compiled execution artifact."""
    raise NotImplementedError


def capture_frontend_error(model, example_inputs, supported_ops):
    """Capture PyTorch to Relax ingestion errors for unsupported ops."""
    raise NotImplementedError
