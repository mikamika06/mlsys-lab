def run_inference(session_spec, inputs, provider="CPUExecutionProvider"):
    raise NotImplementedError


def measure_latency(session_spec, inputs, provider="CPUExecutionProvider", iterations=10):
    raise NotImplementedError
