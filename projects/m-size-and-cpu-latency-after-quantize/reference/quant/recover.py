import numpy as np


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
