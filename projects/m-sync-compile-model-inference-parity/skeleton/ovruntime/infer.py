class InferRequest:
    """Infer request for synchronous or asynchronous execution."""

    def __init__(self, compiled_model):
        raise NotImplementedError

    def infer(self, inputs):
        raise NotImplementedError

    def start_async(self, inputs):
        raise NotImplementedError

    def wait(self):
        raise NotImplementedError


def benchmark_pipeline(compiled_model, inputs_list, mode="sync"):
    """Run inference on a sequence of inputs in sync or async mode and measure latency ticks."""
    raise NotImplementedError
