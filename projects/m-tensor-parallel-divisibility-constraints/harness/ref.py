import random

MARLIN_IN_ALIGN = 128
MARLIN_OUT_ALIGN = 256
MARLIN_SPEED = 10000.0
FALLBACK_SPEED = 2500.0


def gen_layers(seed, n=20):
    rng = random.Random(seed)
    layers = []
    for i in range(n):
        style = rng.choice(["row", "col"])
        in_f = rng.randint(1, 100) * 32
        out_f = rng.randint(1, 100) * 64
        layers.append({"name": f"layer_{i}", "style": style, "in_features": in_f, "out_features": out_f})
    return layers


def _local_shape(layer, tp_size):
    if layer["style"] == "row":
        return layer["in_features"] // tp_size, layer["out_features"]
    return layer["in_features"], layer["out_features"] // tp_size


def check_eligibility(layers, tp_size):
    out = []
    for layer in layers:
        lin, lout = _local_shape(layer, tp_size)
        out.append((lin % MARLIN_IN_ALIGN == 0) and (lout % MARLIN_OUT_ALIGN == 0))
    return out


def evaluate_performance(layers, tp_size):
    elig = check_eligibility(layers, tp_size)
    total = 0.0
    for layer, is_elig in zip(layers, elig):
        lin, lout = _local_shape(layer, tp_size)
        speed = MARLIN_SPEED if is_elig else FALLBACK_SPEED
        total += (lin * lout) / speed
    return {"eligible_count": sum(elig), "total_layers": len(layers), "estimated_time": total}


def pad_for_marlin(layers, tp_size):
    out = []
    for layer in layers:
        in_a = MARLIN_IN_ALIGN * tp_size if layer["style"] == "row" else MARLIN_IN_ALIGN
        out_a = MARLIN_OUT_ALIGN if layer["style"] == "row" else MARLIN_OUT_ALIGN * tp_size
        p_in = (in_a - (layer["in_features"] % in_a)) % in_a
        p_out = (out_a - (layer["out_features"] % out_a)) % out_a
        out.append({
            "name": layer["name"],
            "style": layer["style"],
            "in_features": layer["in_features"] + p_in,
            "out_features": layer["out_features"] + p_out
        })
    return out
