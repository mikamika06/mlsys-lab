import numpy as np


class CompressionPipeline:
    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def prune_then_quantize(self, sparsity: float, bits: int) -> np.ndarray:
        raise NotImplementedError

    def quantize_then_prune(self, sparsity: float, bits: int) -> np.ndarray:
        raise NotImplementedError

    def evaluate_error(self, compressed: np.ndarray) -> float:
        raise NotImplementedError

    def optimal_order(self, sparsity: float, bits: int) -> str:
        raise NotImplementedError

    def measure_gains(self, sparsity: float, bits: int) -> dict:
        raise NotImplementedError

    def transfer_recipe(self, new_weights: np.ndarray, sparsity: float, bits: int) -> np.ndarray:
        raise NotImplementedError
