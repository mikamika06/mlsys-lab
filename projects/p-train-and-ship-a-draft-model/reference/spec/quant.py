import numpy as np
from spec.draft import DraftModel

def quantize_weights(w):
    w_min = np.min(w)
    w_max = np.max(w)
    if w_max == w_min:
        return np.zeros_like(w, dtype=np.int8), w_min, 1.0
    scale = 255.0 / (w_max - w_min)
    w_q = np.clip(np.round((w - w_min) * scale), 0, 255) - 128
    return w_q.astype(np.int8), w_min, scale

def dequantize_weights(w_q, w_min, scale):
    return (w_q.astype(np.float32) + 128.0) / scale + w_min

class QuantizedDraft:
    def __init__(self, draft: DraftModel):
        self.vocab_size = draft.vocab_size
        self.w1_q, self.w1_min, self.w1_scale = quantize_weights(draft.W1)
        self.w2_q, self.w2_min, self.w2_scale = quantize_weights(draft.W2)

    def get_probs(self, token: int):
        w1 = dequantize_weights(self.w1_q, self.w1_min, self.w1_scale)
        w2 = dequantize_weights(self.w2_q, self.w2_min, self.w2_scale)
        h = w1[token]
        logits = h @ w2
        logits -= np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)
