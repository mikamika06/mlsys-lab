import numpy as np


class CompressionPipeline:
    def __init__(self, weights: np.ndarray):
        self.weights = weights.astype(np.float32)

    def prune_then_quantize(self, sparsity: float, bits: int) -> np.ndarray:
        w = self.weights.copy()
        flat = np.abs(w)
        thresh = np.percentile(flat, sparsity * 100)
        mask = np.abs(w) >= thresh
        w_pruned = w * mask
        levels = 2 ** bits - 1
        max_val = np.max(np.abs(w_pruned))
        if max_val == 0:
            return w_pruned
        scale = max_val / levels
        w_quant = np.round(w_pruned / scale) * scale
        return w_quant

    def quantize_then_prune(self, sparsity: float, bits: int) -> np.ndarray:
        w = self.weights.copy()
        levels = 2 ** bits - 1
        max_val = np.max(np.abs(w))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / levels
        w_quant = np.round(w / scale) * scale
        flat = np.abs(w_quant)
        thresh = np.percentile(flat, sparsity * 100)
        mask = np.abs(w_quant) >= thresh
        w_final = w_quant * mask
        return w_final

    def evaluate_error(self, compressed: np.ndarray) -> float:
        return float(np.mean((self.weights - compressed) ** 2))

    def optimal_order(self, sparsity: float, bits: int) -> str:
        ptq = self.evaluate_error(self.prune_then_quantize(sparsity, bits))
        qtp = self.evaluate_error(self.quantize_then_prune(sparsity, bits))
        return "prune_first" if ptq <= qtp else "quant_first"

    def measure_gains(self, sparsity: float, bits: int) -> dict:
        return {
            "size_ratio": (1.0 - sparsity) * (bits / 32.0),
            "speedup_estimate": 1.0 / (1.0 - 0.5 * sparsity)
        }

    def transfer_recipe(self, new_weights: np.ndarray, sparsity: float, bits: int) -> np.ndarray:
        p = CompressionPipeline(new_weights)
        order = self.optimal_order(sparsity, bits)
        if order == "prune_first":
            return p.prune_then_quantize(sparsity, bits)
        return p.quantize_then_prune(sparsity, bits)
