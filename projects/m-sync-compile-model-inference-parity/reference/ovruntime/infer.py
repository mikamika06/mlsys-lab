import numpy as np


class InferRequest:
    """Infer request for synchronous or asynchronous execution."""

    def __init__(self, compiled_model):
        self.compiled_model = compiled_model
        self.output = None
        self._pending_input = None
        self.latency_ticks = 0

    def infer(self, inputs):
        inputs = np.asarray(inputs, dtype=np.float32)
        expected = self.compiled_model.input_shape
        if inputs.shape != expected:
            raise ValueError(
                f"Shape mismatch: expected {expected}, got {inputs.shape}"
            )

        ticks = 10
        x = inputs
        for layer in self.compiled_model.config["layers"]:
            x = np.dot(x, layer["weights"]) + layer["bias"]
            if layer.get("activation") == "relu":
                x = np.maximum(0.0, x)
            ticks += 5

        self.output = x
        self.latency_ticks = ticks
        return self.output

    def start_async(self, inputs):
        inputs = np.asarray(inputs, dtype=np.float32)
        expected = self.compiled_model.input_shape
        if inputs.shape != expected:
            raise ValueError(
                f"Shape mismatch: expected {expected}, got {inputs.shape}"
            )
        self._pending_input = inputs

    def wait(self):
        if self._pending_input is None:
            return self.output

        ticks = 2
        x = self._pending_input
        for layer in self.compiled_model.config["layers"]:
            x = np.dot(x, layer["weights"]) + layer["bias"]
            if layer.get("activation") == "relu":
                x = np.maximum(0.0, x)
            ticks += 5

        self.output = x
        self.latency_ticks = ticks
        self._pending_input = None
        return self.output


def benchmark_pipeline(compiled_model, inputs_list, mode="sync"):
    """Run inference on a sequence of inputs in sync or async mode and measure latency ticks."""
    reqs = [compiled_model.create_infer_request() for _ in inputs_list]
    total_ticks = 0
    results = []

    if mode == "sync":
        for req, inp in zip(reqs, inputs_list):
            out = req.infer(inp)
            total_ticks += req.latency_ticks
            results.append(out)
    elif mode == "async":
        for req, inp in zip(reqs, inputs_list):
            req.start_async(inp)
        for req in reqs:
            out = req.wait()
            total_ticks += req.latency_ticks
            results.append(out)
        total_ticks = int(total_ticks * 0.7)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return results, total_ticks
