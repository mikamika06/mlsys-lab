def compute_performance_metrics(runs):
    ov = [r["tokens_per_sec"] for r in runs if r["engine"] == "openvino"]
    ort = [r["tokens_per_sec"] for r in runs if r["engine"] == "onnxruntime"]
    avg_ov = sum(ov) / len(ov) if ov else 0.0
    avg_ort = sum(ort) / len(ort) if ort else 0.0
    ratio = avg_ov / avg_ort if avg_ort > 0 else 0.0
    return {"avg_openvino": avg_ov, "avg_onnxruntime": avg_ort, "throughput_ratio": ratio}
