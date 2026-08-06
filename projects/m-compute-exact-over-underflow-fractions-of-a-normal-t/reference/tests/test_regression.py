"""Regression tests for numerical diagnostics module."""

from numdiag.classifier import classify_training_log_symptoms


def test_classifier_precision_handling():
    logs = [
        {"grad_norm": 0.0, "loss": 2.5, "loss_delta": -0.01, "unique_activation_ratio": 1.0},
        {"grad_norm": 50000.0, "loss": float("nan"), "is_nan_or_inf": True},
        {"grad_norm": 1e-6, "loss": 1.2, "loss_delta": 0.0000001, "unique_activation_ratio": 0.99},
        {"grad_norm": 0.1, "loss": 0.5, "loss_delta": -0.05, "unique_activation_ratio": 0.01},
    ]
    expected = [
        "FP16_UNDERFLOW",
        "FP16_OVERFLOW",
        "GRADIENT_VANISHING",
        "REPRESENTATION_COLLAPSE",
    ]
    predictions = classify_training_log_symptoms(logs)
    assert predictions == expected, f"Expected {expected}, got {predictions}"
