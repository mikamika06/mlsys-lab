import math

SCALE_BYTES = 2
NO_QUANT_KINDS = {"embed", "norm"}

SCHEMES = [
    {"name": "W16A16", "bits": 16, "group_size": None},
    {"name": "W8A16", "bits": 8, "group_size": 0},
    {"name": "W8A8", "bits": 8, "group_size": 128},
    {"name": "W4A16", "bits": 4, "group_size": 128},
    {"name": "W4A8", "bits": 4, "group_size": 64},
    {"name": "W4A4", "bits": 4, "group_size": 32},
]

MODEL_TRANSFORMER = {"tensors": [
    {"name": "embed", "kind": "embed", "count": 32000 * 768},
    {"name": "layer0.attn", "kind": "linear", "count": 768 * 768 * 4},
    {"name": "layer0.mlp", "kind": "linear", "count": 768 * 3072 * 2},
    {"name": "layer0.norm", "kind": "norm", "count": 768},
    {"name": "layer1.attn", "kind": "linear", "count": 768 * 768 * 4},
    {"name": "layer1.mlp", "kind": "linear", "count": 768 * 3072 * 2},
    {"name": "layer1.norm", "kind": "norm", "count": 768},
    {"name": "final_norm", "kind": "norm", "count": 768},
]}

MODEL_TINY = {"tensors": [
    {"name": "embed", "kind": "embed", "count": 100},
    {"name": "linear_tiny", "kind": "linear", "count": 5},
    {"name": "linear_mid", "kind": "linear", "count": 1000},
    {"name": "norm", "kind": "norm", "count": 16},
]}

MODEL_ODD = {"tensors": [
    {"name": "big", "kind": "linear", "count": 7000003},
]}

MODEL_NO_LINEAR = {"tensors": [
    {"name": "n1", "kind": "norm", "count": 4096},
    {"name": "n2", "kind": "norm", "count": 4096},
]}

CONFIGS = [
    (MODEL_TRANSFORMER, SCHEMES),
    (MODEL_TINY, SCHEMES),
    (MODEL_ODD, SCHEMES),
    (MODEL_NO_LINEAR, SCHEMES),
]

HARDWARE_OLD = {"name": "old-gpu", "native_bits": [16, 8]}
HARDWARE_NEW = {"name": "new-gpu", "native_bits": [16, 8, 4]}
HARDWARE_FP16_ONLY = {"name": "cpu-fallback", "native_bits": [16]}

CASES = [
    (MODEL_TRANSFORMER, SCHEMES, HARDWARE_OLD),
    (MODEL_TRANSFORMER, SCHEMES, HARDWARE_NEW),
    (MODEL_TINY, SCHEMES, HARDWARE_FP16_ONLY),
    (MODEL_ODD, SCHEMES, HARDWARE_OLD),
    (MODEL_NO_LINEAR, SCHEMES, HARDWARE_NEW),
]


def tensor_bytes(tensor, scheme):
    if tensor["kind"] in NO_QUANT_KINDS or scheme["bits"] >= 16:
        return tensor["count"] * 2
    total_bits = tensor["count"] * scheme["bits"]
    weight_bytes = math.ceil(total_bits / 8)
    group_size = scheme["group_size"]
    if not group_size:
        groups = 1
    else:
        groups = math.ceil(tensor["count"] / group_size)
    return weight_bytes + groups * SCALE_BYTES


def disk_size(model, scheme):
    total = 0
    for tensor in model["tensors"]:
        total += tensor_bytes(tensor, scheme)
    return total


def size_table(model, schemes):
    baseline = disk_size(model, schemes[0])
    table = []
    for scheme in schemes:
        size = disk_size(model, scheme)
        table.append({
            "scheme": scheme["name"],
            "bits": scheme["bits"],
            "bytes": size,
            "ratio": size / baseline,
        })
    return table


def hardware_native(scheme, hardware):
    return scheme["bits"] in hardware["native_bits"]


def gate_table(model, hardware, schemes):
    table = size_table(model, schemes)
    for row, scheme in zip(table, schemes):
        row["native"] = hardware_native(scheme, hardware)
    return table


def best_native_scheme(model, hardware, schemes):
    candidates = [s for s in schemes if hardware_native(s, hardware)]
    if not candidates:
        return None
    winner = candidates[0]
    winner_bytes = disk_size(model, winner)
    for scheme in candidates[1:]:
        size = disk_size(model, scheme)
        if size < winner_bytes:
            winner = scheme
            winner_bytes = size
    return winner["name"]
