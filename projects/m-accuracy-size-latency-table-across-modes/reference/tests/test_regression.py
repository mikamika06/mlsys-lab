import numpy as np

from quanteval.modes import evaluate_mode_output
from quanteval.table import build_ptq_summary_table


def test_full_int8_differs_from_dynamic():
    np.random.seed(42)
    w = np.random.randn(8, 16).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    x = np.random.randn(4, 16).astype(np.float32) * 10.0
    cal = (-1.0, 1.0)

    out_dyn = evaluate_mode_output(w, b, x, "dynamic_int8", cal)
    out_full = evaluate_mode_output(w, b, x, "full_int8", cal)

    diff = float(np.max(np.abs(out_dyn - out_full)))
    assert diff > 1e-3


def test_summary_table_modes():
    np.random.seed(42)
    layers = [{
        "name": "l1",
        "weights": np.random.randn(16, 16).astype(np.float32),
        "bias": np.random.randn(16).astype(np.float32),
        "calibration_range": (-2.0, 2.0)
    }]
    hw = {
        "memory_bandwidth_gbps": 10.0,
        "fp32_tflops": 1.0,
        "fp16_tflops": 2.0,
        "int8_tops": 4.0,
        "dynamic_quant_overhead_us": 5.0
    }
    inputs = [np.random.randn(2, 16).astype(np.float32)]

    table = build_ptq_summary_table(layers, hw, inputs)
    assert table["full_int8"]["size_ratio"] < table["fp32"]["size_ratio"]
    assert table["dynamic_int8"]["latency_us"] != table["full_int8"]["latency_us"]
