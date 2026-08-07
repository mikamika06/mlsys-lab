import numpy as np


def get_test_weights_and_inputs():
    np.random.seed(42)
    weights = np.random.randn(128, 256).astype(np.float32)
    inputs = np.random.randn(8, 256).astype(np.float32)
    return weights, inputs


def get_test_logits(n_runs=10, gamma=4, vocab_size=32):
    np.random.seed(1337)
    target_logits = np.random.randn(n_runs, gamma, vocab_size).astype(np.float32)
    draft_fp16_logits = target_logits + np.random.randn(n_runs, gamma, vocab_size).astype(np.float32) * 0.1
    draft_int8_logits = draft_fp16_logits + np.random.randn(n_runs, gamma, vocab_size).astype(np.float32) * 0.5
    return draft_fp16_logits, draft_int8_logits, target_logits
