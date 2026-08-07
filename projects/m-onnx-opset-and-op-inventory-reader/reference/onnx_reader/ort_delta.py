def compute_ort_conversion_delta(onnx_meta, ort_meta):
    onnx_size = onnx_meta.get("file_size_bytes", 0)
    ort_size = ort_meta.get("file_size_bytes", 0)
    onnx_load = onnx_meta.get("load_time_ms", 0.0)
    ort_load = ort_meta.get("load_time_ms", 0.0)

    size_delta_bytes = ort_size - onnx_size
    size_reduction_ratio = (onnx_size - ort_size) / float(onnx_size) if onnx_size > 0 else 0.0
    load_time_delta_ms = ort_load - onnx_load
    speedup_factor = (onnx_load / ort_load) if ort_load > 0 else 0.0

    return {
        "size_delta_bytes": size_delta_bytes,
        "size_reduction_ratio": round(size_reduction_ratio, 4),
        "load_time_delta_ms": round(load_time_delta_ms, 4),
        "speedup_factor": round(speedup_factor, 4),
    }
