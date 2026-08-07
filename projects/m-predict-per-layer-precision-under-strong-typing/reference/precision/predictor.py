def predict_layer_precision(layer, constraints):
    max_val = layer.get("max_val", 1.0)
    min_val = layer.get("min_val", 0.0)
    dynamic_range = max(abs(max_val), abs(min_val))
    if constraints.get("force_fp32", False):
        return "FP32"
    if dynamic_range > 65504.0:
        return "FP32"
    if constraints.get("prefer_tf32", True) and layer.get("op_type") in ("MatMul", "Convolution"):
        return "TF32"
    if layer.get("allow_fp16", True):
        return "FP16"
    return "FP32"
