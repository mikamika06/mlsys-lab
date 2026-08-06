def compute_mxfp_share(model_spec):
    total_bytes = 0
    mxfp_bytes = 0
    for layer in model_spec.get("layers", []):
        b = layer.get("bytes", 0)
        total_bytes += b
        if layer.get("format") == "mxfp4":
            mxfp_bytes += b
    if total_bytes == 0:
        return 0.0
    return float(mxfp_bytes) / float(total_bytes)
