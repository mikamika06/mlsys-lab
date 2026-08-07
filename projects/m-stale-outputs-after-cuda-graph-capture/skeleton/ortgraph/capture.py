"""CUDA graph capture runner and output freshness validator."""


def simulate_capture_and_run(execution_steps, replay_inputs):
    raise NotImplementedError


def detect_stale_outputs(capture_result):
    raise NotImplementedError
