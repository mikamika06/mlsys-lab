import numpy as np


def compute_rope_inv_freqs(method, head_dim, original_max_len, target_max_len, base=10000.0, scale_factor=1.0, yarn_beta_fast=32.0, yarn_beta_slow=1.0):
    raise NotImplementedError


def evaluate_synthetic_perplexity(logits, targets, inv_freqs, seq_len, scale_factor=1.0, attention_factor=1.0):
    raise NotImplementedError
