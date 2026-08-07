MODELS = [
    {
        "name": "audio_upsampler_v1",
        "ops": [{"type": "conv1d"}, {"type": "add"}, {"type": "mul"}, {"type": "average_pool2d"}],
        "weights": [
            {"name": "enc_conv", "count": 4096, "channels": 32},
            {"name": "dec_conv", "count": 16384, "channels": 64}
        ]
    },
    {
        "name": "super_res_light",
        "ops": [{"type": "conv2d"}, {"type": "depthwise_conv2d"}, {"type": "add"}],
        "weights": [
            {"name": "stem", "count": 2048, "channels": 16},
            {"name": "res1", "count": 8192, "channels": 32},
            {"name": "res2", "count": 8192, "channels": 32}
        ]
    },
    {
        "name": "speech_denoiser",
        "ops": [{"type": "fully_connected"}, {"type": "add"}, {"type": "mul"}],
        "weights": [
            {"name": "fc1", "count": 32768, "channels": 128},
            {"name": "fc2", "count": 16384, "channels": 64}
        ]
    },
    {
        "name": "voice_filter",
        "ops": [{"type": "conv1d"}, {"type": "max_pool2d"}],
        "weights": [
            {"name": "conv_a", "count": 1024, "channels": 8}
        ]
    }
]

HEADER_OVERHEAD = 320
PER_TENSOR_OVERHEAD = 64
ALIGNMENT_BYTES = 16

OP_SUPPORT = {
    "dynamic_range": {"conv1d", "conv2d", "depthwise_conv2d", "fully_connected"},
    "int8": {"conv1d", "conv2d", "depthwise_conv2d", "fully_connected", "add", "mul", "average_pool2d", "max_pool2d"},
    "int16x8": {"conv1d", "conv2d", "depthwise_conv2d", "fully_connected", "add", "average_pool2d"}
}


def _align(size, alignment=ALIGNMENT_BYTES):
    return (size + alignment - 1) & ~(alignment - 1)


def predict_quantized_size(model_spec, quant_mode="int8"):
    bits = 16 if quant_mode == "int16x8" else 8
    return sum((p["count"] * bits) // 8 for p in model_spec["weights"])


def actual_flatbuffer_size(model_spec, quant_mode="int8"):
    bits = 16 if quant_mode == "int16x8" else 8
    total = HEADER_OVERHEAD
    for p in model_spec["weights"]:
        raw = (p["count"] * bits) // 8
        aligned = _align(raw)
        meta = PER_TENSOR_OVERHEAD
        if quant_mode in ("int8", "int16x8"):
            num_channels = p.get("channels", 1)
            meta += num_channels * (8 if quant_mode == "int16x8" else 4)
        total += aligned + meta
    return total


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
