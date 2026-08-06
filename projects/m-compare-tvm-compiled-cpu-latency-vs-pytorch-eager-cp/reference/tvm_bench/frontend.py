class UnsupportedOpError(Exception):
    """Raised when an operator is not supported by the frontend."""

    pass


class DummyCompiledModule:

    def __init__(self, model, ops):
        self.model = model
        self.ops = ops

    def __call__(self, *args):
        return self.model(*args)


def import_and_compile(model, example_inputs, supported_ops):
    """Import PyTorch model to Relax representation and return compiled execution artifact."""
    ops = getattr(model, "ops", [])
    for op in ops:
        if op not in supported_ops:
            raise UnsupportedOpError(f"Unsupported op: {op}")
    return DummyCompiledModule(model, ops)


def capture_frontend_error(model, example_inputs, supported_ops):
    """Capture PyTorch to Relax ingestion errors for unsupported ops."""
    try:
        import_and_compile(model, example_inputs, supported_ops)
    except UnsupportedOpError as e:
        return True, str(e)
    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}: {e}"
    return False, "No error raised"
