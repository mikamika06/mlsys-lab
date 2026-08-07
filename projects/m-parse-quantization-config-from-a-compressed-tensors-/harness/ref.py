import numpy as np

CONFIGS = [
    {
        "quantization_config": {
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "int", "group_size": 128, "symmetric": False},
                    "input_activations": {"num_bits": 16}
                }
            }
        }
    },
    {
        "quantization_config": {
            "format": "nvfp4",
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                    "weight_scales": {"group_size": 0}
                }
            }
        }
    },
    {
        "quantization_config": {
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "int", "group_size": 64, "symmetric": True},
                    "input_activations": {"num_bits": 16}
                }
            }
        }
    }
]


def parse_quant_config(cfg):
    qc = cfg.get("quantization_config", {})
    fmt = qc.get("format", "")
    group = qc.get("config_groups", {}).get("group_0", {})
    w = group.get("weights", {})

    w_bits = w.get("num_bits", 8)
    group_size = w.get("group_size", 0)

    if fmt == "nvfp4":
        has_global = "weight_scales" in group and group["weight_scales"].get("group_size", 1) == 0
        return {
            "format": "nvfp4",
            "w_bits": w_bits,
            "group_size": group_size,
            "global_scale": has_global
        }
    else:
        symmetric = w.get("symmetric", True)
        a_bits = group.get("input_activations", {}).get("num_bits", 16)
        return {
            "format": f"w{w_bits}a{a_bits}",
            "w_bits": w_bits,
            "group_size": group_size,
            "symmetric": symmetric
        }


def dequantize_w4a16(packed_weights, scales, zeros, group_size):
    N, K_half = packed_weights.shape
    K = K_half * 2
    w = np.empty((N, K), dtype=np.uint8)
    w[:, 0::2] = packed_weights & 0x0F
    w[:, 1::2] = (packed_weights >> 4) & 0x0F

    s = np.repeat(scales, group_size, axis=1)
    z = np.repeat(zeros, group_size, axis=1)

    return (w.astype(np.float32) - z) * s


def dequantize_nvfp4(weights, local_scales, global_scale, group_size=16):
    s = np.repeat(local_scales, group_size, axis=1)
    return weights * s * global_scale


def generate_w4a16_fixture():
    np.random.seed(42)
    N, K = 32, 256
    group_size = 64
    packed = np.random.randint(0, 256, size=(N, K // 2), dtype=np.uint8)
    scales = np.random.uniform(0.1, 1.0, size=(N, K // group_size)).astype(np.float32)
    zeros = np.random.randint(0, 15, size=(N, K // group_size)).astype(np.float32)
    return packed, scales, zeros, group_size


def generate_nvfp4_fixture():
    np.random.seed(43)
    N, K = 32, 256
    group_size = 16
    weights = np.random.randn(N, K).astype(np.float32)
    local_scales = np.random.uniform(0.5, 2.0, size=(N, K // group_size)).astype(np.float32)
    global_scale = 0.05
    return weights, local_scales, global_scale, group_size
