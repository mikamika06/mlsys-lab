import hashlib
import os
import time


class EdgeRuntime:
    """Simulated edge model execution engine with compilation caching."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, model_id: str, artifact_bytes: bytes) -> str:
        h = hashlib.sha256(artifact_bytes).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{model_id}_{h}.compiled")

    def compile_model(self, model_id: str, artifact_bytes: bytes) -> bytes:
        time.sleep(0.05)
        header = f"COMPILED:{model_id}:".encode("utf-8")
        return header + hashlib.md5(artifact_bytes).digest()

    def run_inference(self, model_id: str, artifact_bytes: bytes, input_data: list) -> tuple:
        cache_path = self._get_cache_path(model_id, artifact_bytes)
        compile_time = 0.0

        if not os.path.exists(cache_path):
            t0 = time.perf_counter()
            compiled = self.compile_model(model_id, artifact_bytes)
            compile_time = (time.perf_counter() - t0) * 1000.0
            with open(cache_path, "wb") as f:
                f.write(compiled)
        else:
            with open(cache_path, "rb") as f:
                compiled = f.read()

        base_lat = sum(input_data) * 0.1 + len(artifact_bytes) * 0.001
        total_latency = compile_time + base_lat
        output = [x * 2 for x in input_data]
        return output, total_latency
