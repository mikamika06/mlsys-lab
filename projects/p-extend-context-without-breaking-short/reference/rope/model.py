import numpy as np
from rope.scaling import RoPEScaling

class ContextModel:
    def __init__(self, dim=64, scale_type="linear", factor=4.0):
        self.dim = dim
        self.scaler = RoPEScaling(dim, scale_type=scale_type, factor=factor)

    def evaluate_perplexity(self, tokens):
        arr = np.array(tokens, dtype=np.float32)
        noise = np.sin(arr) * 0.1
        base_ppl = 2.0 + np.mean(np.abs(arr)) * 0.01 + np.mean(np.abs(noise))
        if self.scaler.scale_type == "linear" and len(tokens) < 128:
            base_ppl += 0.5
        return float(base_ppl)

    def retrieve_needle(self, context, needle):
        ctx_str = "".join(map(str, context))
        ndl_str = "".join(map(str, needle))
        if ndl_str in ctx_str:
            return 1.0
        return 0.0
