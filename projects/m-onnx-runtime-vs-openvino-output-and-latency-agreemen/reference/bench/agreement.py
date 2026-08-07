import numpy as np


def compute_output_agreement(ort_outputs, ov_outputs, rtol=1e-3, atol=1e-5):
    """Checks numerical agreement and calculates maximum relative error."""
    diff = np.abs(ort_outputs - ov_outputs)
    rel_err = diff / (np.abs(ov_outputs) + 1e-8)
    max_rel_err = float(np.max(rel_err))
    is_agreed = bool(np.allclose(ort_outputs, ov_outputs, rtol=rtol, atol=atol))
    return {"agreed": is_agreed, "max_rel_err": max_rel_err}


def evaluate_mac_runs(mac_records):
    """Evaluates ONNX Runtime vs OpenVINO latency and output agreement on Mac."""
    results = []
    for rec in mac_records:
        agr = compute_output_agreement(rec["ort_out"], rec["ov_out"])
        ort_lat = float(np.median(rec["ort_times_ms"]))
        ov_lat = float(np.median(rec["ov_times_ms"]))
        ratio = ort_lat / (ov_lat + 1e-8)
        results.append({
            "model_id": rec["model_id"],
            "agreed": agr["agreed"],
            "max_rel_err": agr["max_rel_err"],
            "ort_latency_ms": ort_lat,
            "ov_latency_ms": ov_lat,
            "latency_ratio_ort_over_ov": ratio,
        })
    return results
