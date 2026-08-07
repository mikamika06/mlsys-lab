BLOCK_SIZE = 256

KQUANT_BITS = {
    "Q4_K": {"attn": 4.5, "ffn_gate": 4.5, "ffn_down": 3.5, "norm": 16.0, "default": 4.5},
    "Q5_K": {"attn": 5.5, "ffn_gate": 5.5, "ffn_down": 4.5, "norm": 16.0, "default": 5.5},
    "Q6_K": {"attn": 6.5, "ffn_gate": 6.5, "ffn_down": 5.5, "norm": 16.0, "default": 6.5},
    "FP16": {"attn": 16.0, "ffn_gate": 16.0, "ffn_down": 16.0, "norm": 16.0, "default": 16.0},
}


def get_layer_bits(quant_type, role):
    q_map = KQUANT_BITS.get(quant_type, KQUANT_BITS["FP16"])
    return q_map.get(role, q_map["default"])


def calculate_weight_bytes(config):
    num_layers = config["num_layers"]
    hidden_dim = config["hidden_dim"]
    intermediate_dim = config["intermediate_dim"]
    num_heads = config["num_heads"]
    num_kv_heads = config.get("num_kv_heads", num_heads)
    head_dim = hidden_dim // num_heads
    quant_type = config.get("quant_type", "Q4_K")

    q_bits = get_layer_bits(quant_type, "attn")
    k_bits = get_layer_bits(quant_type, "attn")
    v_bits = get_layer_bits(quant_type, "attn")
    o_bits = get_layer_bits(quant_type, "attn")

    gate_bits = get_layer_bits(quant_type, "ffn_gate")
    up_bits = get_layer_bits(quant_type, "ffn_gate")
    down_bits = get_layer_bits(quant_type, "ffn_down")
    norm_bits = get_layer_bits(quant_type, "norm")

    layer_bytes = 0.0

    q_params = hidden_dim * (num_heads * head_dim)
    k_params = hidden_dim * (num_kv_heads * head_dim)
    v_params = hidden_dim * (num_kv_heads * head_dim)
    o_params = (num_heads * head_dim) * hidden_dim

    layer_bytes += (q_params * q_bits) / 8.0
    layer_bytes += (k_params * k_bits) / 8.0
    layer_bytes += (v_params * v_bits) / 8.0
    layer_bytes += (o_params * o_bits) / 8.0

    gate_params = hidden_dim * intermediate_dim
    up_params = hidden_dim * intermediate_dim
    down_params = intermediate_dim * hidden_dim

    layer_bytes += (gate_params * gate_bits) / 8.0
    layer_bytes += (up_params * up_bits) / 8.0
    layer_bytes += (down_params * down_bits) / 8.0

    norm_params = 2 * hidden_dim
    layer_bytes += (norm_params * norm_bits) / 8.0

    total_weight_bytes = layer_bytes * num_layers

    vocab_size = config.get("vocab_size", 32000)
    embed_bits = get_layer_bits(quant_type, "default")
    embed_bytes = (vocab_size * hidden_dim * embed_bits) / 8.0
    total_weight_bytes += embed_bytes * 2.0

    return total_weight_bytes


def calculate_kv_cache_bytes(config, seq_len, batch_size=1):
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    num_kv_heads = config.get("num_kv_heads", num_heads)
    hidden_dim = config["hidden_dim"]
    head_dim = hidden_dim // num_heads
    kv_bytes_per_elem = config.get("kv_bytes_per_elem", 2)

    total_kv_elements = 2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim
    return total_kv_elements * kv_bytes_per_elem


def calculate_activation_bytes(config, seq_len, batch_size=1):
    hidden_dim = config["hidden_dim"]
    intermediate_dim = config["intermediate_dim"]
    act_bytes_per_elem = config.get("act_bytes_per_elem", 2)

    attn_act = batch_size * seq_len * hidden_dim * 4
    ffn_act = batch_size * seq_len * intermediate_dim * 2
    return (attn_act + ffn_act) * act_bytes_per_elem


def predict_resident_vram(config, seq_len, batch_size=1):
    w_bytes = calculate_weight_bytes(config)
    kv_bytes = calculate_kv_cache_bytes(config, seq_len, batch_size)
    act_bytes = calculate_activation_bytes(config, seq_len, batch_size)
    return w_bytes + kv_bytes + act_bytes
