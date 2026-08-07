def calibrate_w8a8(tensors):
    scales = {}
    zero_points = {}
    for name, t in tensors.items():
        min_val = min(t)
        max_val = max(t)
        scale = (max_val - min_val) / 255.0 if max_val != min_val else 1.0
        zp = int(round(-min_val / scale)) if scale != 0 else 0
        zp = max(0, min(255, zp))
        scales[name] = scale
        zero_points[name] = zp
    return {"scales": scales, "zero_points": zero_points}
