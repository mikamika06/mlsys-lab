from memrunner.predictor import KQUANT_BITS, get_layer_bits


def evaluate_kquant_layer_bits(layer_type, quant_type):
    return get_layer_bits(quant_type, layer_type)


def explain_kquant_precision_mix(config):
    quant_type = config.get("quant_type", "Q4_K")
    attn_bits = get_layer_bits(quant_type, "attn")
    gate_bits = get_layer_bits(quant_type, "ffn_gate")
    down_bits = get_layer_bits(quant_type, "ffn_down")

    mixed = (attn_bits != down_bits) or (gate_bits != down_bits)

    return {
        "quant_type": quant_type,
        "attn_bits": attn_bits,
        "ffn_gate_bits": gate_bits,
        "ffn_down_bits": down_bits,
        "is_mixed_precision": mixed,
        "rationale": (
            "K-quants assign higher precision to attention projections and FFN gate "
            "tensors to preserve key features, while aggressively quantizing down projections."
        ),
    }
