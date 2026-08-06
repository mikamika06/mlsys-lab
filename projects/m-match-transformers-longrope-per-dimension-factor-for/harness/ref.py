import numpy as np
from longrope.scaling import compute_longrope_factors as _ref_compute_longrope_factors
from longrope.perplexity import (
    compute_rope_inv_freqs as _ref_compute_rope_inv_freqs,
    evaluate_synthetic_perplexity as _ref_evaluate_synthetic_perplexity,
)
from longrope.entropy import measure_yarn_attention_entropy as _ref_measure_yarn_attention_entropy


def get_test_configs():
    return [
        {"head_dim": 64, "orig_len": 4096, "target_len": 16384, "base": 10000.0},
        {"head_dim": 128, "orig_len": 2048, "target_len": 8192, "base": 50000.0},
        {"head_dim": 32, "orig_len": 8192, "target_len": 32768, "base": 10000.0},
    ]


def oracle_longrope_factors(head_dim, orig_len, target_len, base=10000.0):
    return _ref_compute_longrope_factors(head_dim, orig_len, target_len, base=base)


def oracle_inv_freqs(method, head_dim, orig_len, target_len, base=10000.0, scale_factor=1.0):
    return _ref_compute_rope_inv_freqs(method, head_dim, orig_len, target_len, base=base, scale_factor=scale_factor)


def oracle_perplexity(logits, targets, inv_freqs, seq_len, scale_factor=1.0, attention_factor=1.0):
    return _ref_evaluate_synthetic_perplexity(logits, targets, inv_freqs, seq_len, scale_factor, attention_factor)


def oracle_entropy(q, k, inv_freqs, attention_factor=1.0):
    return _ref_measure_yarn_attention_entropy(q, k, inv_freqs, attention_factor)
