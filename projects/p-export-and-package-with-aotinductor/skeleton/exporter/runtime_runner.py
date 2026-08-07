import numpy as np
from typing import Dict, Any


class StandaloneAOTRunner:
    def __init__(self, artifact_path: str):
        raise NotImplementedError

    def run(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


def benchmark_aot_runner(runner: StandaloneAOTRunner, num_runs: int = 50) -> Dict[str, float]:
    raise NotImplementedError
