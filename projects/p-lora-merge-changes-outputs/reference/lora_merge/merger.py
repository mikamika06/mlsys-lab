import numpy as np


class LoRAMerger:
    def __init__(self, base_weights, lora_A, lora_B, alpha, rank):
        self.base_weights = [np.array(w, dtype=np.float32) for w in base_weights]
        self.lora_A = [np.array(a, dtype=np.float32) for a in lora_A]
        self.lora_B = [np.array(b, dtype=np.float32) for b in lora_B]
        self.alpha = float(alpha)
        self.rank = int(rank)
        self.merged_weights = [w.copy() for w in self.base_weights]

    def measure_layer_diffs(self, x):
        diffs = []
        scaling = self.alpha / self.rank
        for w, a, b in zip(self.base_weights, self.lora_A, self.lora_B):
            delta = (b @ a) * scaling
            naive_merged = w + delta
            out_orig = (w @ x) + (delta @ x)
            out_merged = naive_merged @ x
            diffs.append(float(np.max(np.abs(out_orig - out_merged))))
        return diffs

    def fix_dtype(self):
        scaling = self.alpha / self.rank
        fixed = []
        for w, a, b in zip(self.base_weights, self.lora_A, self.lora_B):
            delta = (b @ a) * scaling
            merged = w.astype(np.float32) + delta.astype(np.float32)
            fixed.append(merged.astype(w.dtype))
        self.merged_weights = fixed
        return True

    def verify_scaling(self):
        return self.rank > 0 and self.alpha > 0

    def safe_merge(self):
        scaling = self.alpha / self.rank
        fixed = []
        for w, a, b in zip(self.base_weights, self.lora_A, self.lora_B):
            delta = (b @ a) * scaling
            merged = w + delta
            fixed.append(merged)
        self.merged_weights = fixed
        return self.merged_weights

    def evaluate_prompts(self, prompts):
        scaling = self.alpha / self.rank
        errors = []
        for p in prompts:
            x = np.array(p, dtype=np.float32)
            out_base = x.copy()
            out_adapter = x.copy()
            for w, a, b in zip(self.base_weights, self.lora_A, self.lora_B):
                out_base = (self.merged_weights[0] @ out_base)
            for w, a, b in zip(self.base_weights, self.lora_A, self.lora_B):
                delta = (b @ a) * scaling
                out_adapter = ((w + delta) @ out_adapter)
            errors.append(float(np.max(np.abs(out_adapter - out_base))))
        return max(errors) if errors else 0.0
