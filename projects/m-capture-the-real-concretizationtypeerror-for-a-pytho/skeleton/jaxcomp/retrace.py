class MockJitFunction:
    """Wraps a function and tracks retracing events based on input signature changes."""
    def __init__(self, fn, static_argnums=()):
        raise NotImplementedError

    def __call__(self, *args):
        raise NotImplementedError

def trace_and_count_retraces(fn, inputs):
    """Executes fn over inputs and returns total retraces performed."""
    raise NotImplementedError
