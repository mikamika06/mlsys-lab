OP_SUPPORT = {
    "dynamic_range": {"conv1d", "conv2d", "depthwise_conv2d", "fully_connected"},
    "int8": {"conv1d", "conv2d", "depthwise_conv2d", "fully_connected", "add", "mul", "average_pool2d", "max_pool2d"},
    "int16x8": {"conv1d", "conv2d", "depthwise_conv2d", "fully_connected", "add", "average_pool2d"}
}


def analyze_op_compatibility(model_spec, quant_mode="int8"):
    supported = OP_SUPPORT.get(quant_mode, set())
    res = []
    for op in model_spec["ops"]:
        is_supp = op["type"] in supported
        req_calib = quant_mode in ("int8", "int16x8")
        is_float_exec = quant_mode == "dynamic_range" and op["type"] not in OP_SUPPORT["dynamic_range"]
        res.append({
            "type": op["type"],
            "supported": is_supp,
            "requires_calibration": req_calib,
            "executes_float": is_float_exec
        })
    return res


def compare_int8_vs_int16x8(model_spec):
    int8_size = sum((w["count"] * 8) // 8 for w in model_spec["weights"])
    int16_size = sum((w["count"] * 16) // 8 for w in model_spec["weights"])
    int8_ops = sum(1 for op in model_spec["ops"] if op["type"] in OP_SUPPORT["int8"])
    int16_ops = sum(1 for op in model_spec["ops"] if op["type"] in OP_SUPPORT["int16x8"])
    return {
        "int8": {"weight_bytes": int8_size, "supported_ops": int8_ops},
        "int16x8": {"weight_bytes": int16_size, "supported_ops": int16_ops},
        "size_ratio": float(int16_size) / float(int8_size) if int8_size > 0 else 1.0
    }
