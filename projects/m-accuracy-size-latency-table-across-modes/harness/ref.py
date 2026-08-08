import numpy as np

MODES = ["FP32", "FP16", "INT8_DYNAMIC", "INT8_FULL"]

RAW_RECORDS = [
    {"mode": "FP32", "size_bytes": 40000000, "latency_ms": 25.0, "accuracy": 0.785},
    {"mode": "FP16", "size_bytes": 20000000, "latency_ms": 18.0, "accuracy": 0.784},
    {"mode": "INT8_DYNAMIC", "size_bytes": 10500000, "latency_ms": 15.0, "accuracy": 0.772},
    {"mode": "INT8_FULL", "size_bytes": 10000000, "latency_ms": 11.0, "accuracy": 0.768},
]


def build_profiles(records):
    out = []
    for r in records:
        out.append({
            "mode": r["mode"],
            "size_bytes": int(r["size_bytes"]),
            "latency_ms": float(r["latency_ms"]),
            "accuracy": float(r["accuracy"])
        })
    return out


def compute_size_ratios(profiles):
    fp32_size = next(p["size_bytes"] for p in profiles if p["mode"] == "FP32")
    out = {}
    for p in profiles:
        out[p["mode"]] = round(p["size_bytes"] / fp32_size, 4)
    return out


def evaluate_tradeoffs(profiles):
    ratios = compute_size_ratios(profiles)
    valid = True
    for p in profiles:
        if p["mode"] != "FP32" and ratios[p["mode"]] >= 1.0:
            valid = False
    return {"valid": valid, "ratios": ratios}
