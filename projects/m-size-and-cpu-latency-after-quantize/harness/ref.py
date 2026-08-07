import numpy as np


def calculate_model_bytes(model_dict):
    total = 0
    for key, info in model_dict.items():
        if isinstance(info, np.ndarray):
            total += info.nbytes
        elif isinstance(info, dict):
            if "bytes" in info:
                total += info["bytes"]
            else:
                shape = info.get("shape", ())
                nelem = 1
                for d in shape:
                    nelem *= d
                dtype = info.get("dtype", "float32")
                if dtype == "float32":
                    b = nelem * 4
                elif dtype in ("float16", "bfloat16"):
                    b = nelem * 2
                elif dtype == "int8":
                    b = nelem * 1
                elif dtype == "int4":
                    b = int(np.ceil(nelem / 2.0))
                else:
                    b = nelem * 4
                b += info.get("extra_bytes", 0)
                total += b
    return total


def profile_quantization(before_model, after_model, bench_fn):
    lat_before = float(bench_fn(before_model))
    lat_after = float(bench_fn(after_model))
    bytes_before = calculate_model_bytes(before_model)
    bytes_after = calculate_model_bytes(after_model)
    ratio = float(bytes_after / bytes_before) if bytes_before > 0 else 1.0
    speedup = float(lat_before / lat_after) if lat_after > 0 else 1.0
    return {
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "size_ratio": ratio,
        "latency_before_ms": lat_before,
        "latency_after_ms": lat_after,
        "speedup": speedup,
    }


def diagnose_noop_layers(before_model, after_model):
    noops = []
    for layer_name, b_info in before_model.items():
        if layer_name not in after_model:
            continue
        a_info = after_model[layer_name]
        b_dtype = b_info.get("dtype") if isinstance(b_info, dict) else "float32"
        a_dtype = a_info.get("dtype") if isinstance(a_info, dict) else "float32"
        is_quantized = (
            a_info.get("quantized", True) if isinstance(a_info, dict) else True
        )

        if not is_quantized or a_dtype == b_dtype:
            noops.append(layer_name)
    return sorted(noops)


def recover_fp32_layer(packed_int4, scale, zero_point, shape, group_size):
    out_feat, in_feat = shape
    unpacked = np.zeros((out_feat, in_feat), dtype=np.float32)
    unpacked[:, 0::2] = (packed_int4 & 0x0F).astype(np.float32)
    unpacked[:, 1::2] = ((packed_int4 >> 4) & 0x0F).astype(np.float32)

    zp = 0.0 if zero_point is None else zero_point.astype(np.float32)
    scale = scale.astype(np.float32)

    scale_exp = np.repeat(scale, group_size, axis=1)
    if isinstance(zp, np.ndarray):
        zp_exp = np.repeat(zp, group_size, axis=1)
    else:
        zp_exp = zp

    return (unpacked - zp_exp) * scale_exp


def recover_state_dict(state_dict):
    out = {}
    prefixes = set()
    for k in state_dict:
        if "." in k:
            prefix = k.rsplit(".", 1)[0]
            prefixes.add(prefix)
        else:
            prefixes.add(k)

    processed_prefixes = set()
    for prefix in sorted(prefixes):
        pk = f"{prefix}.weight_packed"
        if pk in state_dict:
            packed = state_dict[pk]
            scale = state_dict[f"{prefix}.weight_scale"]
            zp = state_dict.get(f"{prefix}.weight_zero_point", None)
            shape = state_dict[f"{prefix}.shape"]
            group_size = state_dict[f"{prefix}.group_size"]
            w_fp32 = recover_fp32_layer(packed, scale, zp, shape, group_size)
            out[f"{prefix}.weight"] = w_fp32
            processed_prefixes.add(prefix)

    for k, v in state_dict.items():
        prefix = k.rsplit(".", 1)[0] if "." in k else k
        if prefix not in processed_prefixes:
            if k.endswith(".weight") or not any(
                k.startswith(p + ".") for p in processed_prefixes
            ):
                out[k] = v

    return out


np.random.seed(42)

BEFORE_MODEL_1 = {
    "layer1.weight": {"dtype": "float32", "shape": (128, 128), "quantized": False},
    "layer2.weight": {"dtype": "float32", "shape": (256, 128), "quantized": False},
    "layer3.weight": {"dtype": "float32", "shape": (64, 64), "quantized": False},
}

AFTER_MODEL_1 = {
    "layer1.weight": {
        "dtype": "int4",
        "shape": (128, 128),
        "quantized": True,
        "extra_bytes": 1024,
    },
    "layer2.weight": {
        "dtype": "int4",
        "shape": (256, 128),
        "quantized": True,
        "extra_bytes": 2048,
    },
    "layer3.weight": {"dtype": "float32", "shape": (64, 64), "quantized": False},
}


def BENCH_FN_1(model):
    has_int4 = any(
        isinstance(v, dict) and v.get("dtype") == "int4" for v in model.values()
    )
    return 12.5 if has_int4 else 50.0


def make_sample_state_dict():
    shape = (16, 32)
    group_size = 16
    out_feat, in_feat = shape
    num_groups = in_feat // group_size

    np.random.seed(123)
    raw_int4 = np.random.randint(0, 16, size=(out_feat, in_feat), dtype=np.uint8)
    packed = (raw_int4[:, 0::2] & 0x0F) | ((raw_int4[:, 1::2] & 0x0F) << 4)

    scale = np.random.uniform(0.01, 0.1, size=(out_feat, num_groups)).astype(
        np.float32
    )
    zp = np.random.randint(0, 8, size=(out_feat, num_groups)).astype(np.float32)

    scale_exp = np.repeat(scale, group_size, axis=1)
    zp_exp = np.repeat(zp, group_size, axis=1)
    expected_fp32 = (raw_int4.astype(np.float32) - zp_exp) * scale_exp

    unquant_weight = np.random.randn(8, 8).astype(np.float32)

    sd = {
        "encoder.block0.weight_packed": packed,
        "encoder.block0.weight_scale": scale,
        "encoder.block0.weight_zero_point": zp,
        "encoder.block0.shape": shape,
        "encoder.block0.group_size": group_size,
        "head.weight": unquant_weight,
    }

    expected_recovered = {
        "encoder.block0.weight": expected_fp32,
        "head.weight": unquant_weight,
    }

    return sd, expected_recovered


SAMPLE_STATE_DICT, SAMPLE_EXPECTED_RECOVERED = make_sample_state_dict()
