"""Regression tests for backend classifier and failure explainer."""

import sys
sys.path.insert(0, ".")

from fa_backend.classifier import classify_backend, is_backend_supported
from fa_backend.failure import explain_platform_failure


def test_classifier_prefers_hopper_on_sm90():
    mcfg = {"compute_capability": (9, 0), "smem_per_sm_bytes": 100000, "sm_count": 132}
    ispec = {"head_dim": 128, "dtype": "bfloat16", "seq_len": 2048, "num_heads": 32}
    backend = classify_backend(mcfg, ispec)
    assert backend == "FA3_HOPPER", f"Expected FA3_HOPPER, got {backend}"


def test_classifier_falls_back_on_sm80():
    mcfg = {"compute_capability": (8, 0), "smem_per_sm_bytes": 65536, "sm_count": 108}
    ispec = {"head_dim": 128, "dtype": "float16", "seq_len": 2048, "num_heads": 32}
    backend = classify_backend(mcfg, ispec)
    assert backend == "FA2_CUDA", f"Expected FA2_CUDA, got {backend}"


def test_failure_explanation_for_old_cc():
    mcfg = {"compute_capability": (7, 0), "smem_per_sm_bytes": 49152, "sm_count": 80}
    ispec = {"head_dim": 128, "dtype": "float16", "seq_len": 1024, "num_heads": 16}
    reason = explain_platform_failure("FA3_HOPPER", mcfg, ispec)
    assert "UNSUPPORTED_COMPUTE_CAPABILITY" in reason, f"Unexpected reason: {reason}"


def test_failure_explanation_unaligned_dim():
    mcfg = {"compute_capability": (9, 0), "smem_per_sm_bytes": 100000, "sm_count": 132}
    ispec = {"head_dim": 100, "dtype": "float16", "seq_len": 1024, "num_heads": 16}
    reason = explain_platform_failure("FA3_HOPPER", mcfg, ispec)
    assert "INVALID_HEAD_DIM_ALIGNMENT" in reason, f"Unexpected reason: {reason}"
