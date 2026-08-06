import os


class EdgeRuntime:
    """Simulated edge model execution engine with compilation caching."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def compile_model(self, model_id: str, artifact_bytes: bytes) -> bytes:
        """Simulate expensive compilation step."""
        raise NotImplementedError

    def run_inference(self, model_id: str, artifact_bytes: bytes, input_data: list) -> tuple:
        """Run model inference returning (output_data, latency_ms)."""
        raise NotImplementedError
