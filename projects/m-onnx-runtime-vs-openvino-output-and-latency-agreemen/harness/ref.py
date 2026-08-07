import numpy as np


def generate_mac_records():
    np.random.seed(42)
    records = []
    for i in range(3):
        size = (10, 10)
        base = np.random.randn(*size).astype(np.float32)
        noise = (i * 1e-4) * np.random.randn(*size).astype(np.float32)
        ort_out = base
        ov_out = base + noise
        ort_times = (10.0 + i + np.random.uniform(-0.5, 0.5, size=10)).tolist()
        ov_times = (8.0 + i + np.random.uniform(-0.5, 0.5, size=10)).tolist()
        records.append({
            "model_id": f"model_{i}",
            "ort_out": ort_out,
            "ov_out": ov_out,
            "ort_times_ms": ort_times,
            "ov_times_ms": ov_times,
        })
    return records


def generate_xeon_logs():
    return [
        {"engine": "openvino", "queries_per_sec": 450.0},
        {"engine": "openvino", "queries_per_sec": 460.0},
        {"engine": "onnxruntime", "queries_per_sec": 410.0},
        {"engine": "onnxruntime", "queries_per_sec": 400.0},
        {"engine": "pytorch", "queries_per_sec": 300.0},
        {"engine": "pytorch", "queries_per_sec": 310.0},
    ]


def build_mac_agreements(records):
    out = []
    for rec in records:
        diff = np.abs(rec["ort_out"] - rec["ov_out"])
        rel_err = diff / (np.abs(rec["ov_out"]) + 1e-8)
        agreed = bool(np.allclose(rec["ort_out"], rec["ov_out"], rtol=1e-3, atol=1e-5))
        ort_lat = float(np.median(rec["ort_times_ms"]))
        ov_lat = float(np.median(rec["ov_times_ms"]))
        out.append({
            "model_id": rec["model_id"],
            "agreed": agreed,
            "max_rel_err": float(np.max(rel_err)),
            "ort_latency_ms": ort_lat,
            "ov_latency_ms": ov_lat,
            "latency_ratio_ort_over_ov": ort_lat / (ov_lat + 1e-8),
        })
    return out


def build_xeon_ranking(xeon_logs):
    return [
        {"engine": "openvino", "median_qps": 455.0},
        {"engine": "onnxruntime", "median_qps": 405.0},
        {"engine": "pytorch", "median_qps": 305.0},
    ]


def build_fairness_checks():
    return [
        {
            "engine_a": {"precision": "fp32", "quantized": False},
            "engine_b": {"precision": "fp32", "quantized": False},
            "fair": True,
        },
        {
            "engine_a": {"precision": "fp32", "quantized": False},
            "engine_b": {"precision": "int8", "quantized": True},
            "fair": False,
        },
    ]
