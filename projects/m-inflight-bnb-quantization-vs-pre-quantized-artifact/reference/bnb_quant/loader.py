import numpy as np


def quantize_fp16_to_int8(weight_fp16):
    """Dynamically quantizes FP16 weight array into Int8 with per-row scaling."""
    abs_max = np.abs(weight_fp16).max(axis=-1, keepdims=True)
    abs_max = np.maximum(abs_max, 1e-8)
    scales = abs_max / 127.0
    quantized = np.round(weight_fp16 / scales).astype(np.int8)
    return quantized, scales.astype(np.float32)


def pack_prequantized_artifact(weight_fp16):
    """Packs pre-quantized weights into an offline static artifact."""
    q_weight, scales = quantize_fp16_to_int8(weight_fp16)
    return {
        "qweight": q_weight,
        "scales": scales,
        "packed": True
    }


def load_model_weights(raw_weights, mode="inflight"):
    """Loads weights either via dynamic inflight quantization or pre-quantized artifact restore."""
    loaded = {}
    if mode == "inflight":
        for name, weight in raw_weights.items():
            q_w, scales = quantize_fp16_to_int8(weight)
            loaded[name] = {"qweight": q_w, "scales": scales, "mode": "inflight"}
    elif mode == "prequantized":
        for name, weight in raw_weights.items():
            artifact = pack_prequantized_artifact(weight)
            loaded[name] = artifact
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return loaded
