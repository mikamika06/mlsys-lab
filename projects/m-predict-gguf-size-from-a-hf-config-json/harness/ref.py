import math

CONFIGS = [
    {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
        "tie_word_embeddings": False
    },
    {
        "architectures": ["MixtralForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "tie_word_embeddings": False
    },
    {
        "architectures": ["GemmaForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 16384,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "vocab_size": 256000,
        "tie_word_embeddings": True
    }
]

def get_tensor_counts(config):
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]

    tensors = {}

    tensors["token_embd.weight"] = (vocab, hidden)
    if not config.get("tie_word_embeddings", False):
        tensors["output.weight"] = (vocab, hidden)

    tensors["output_norm.weight"] = (hidden,)

    q_size = hidden * (num_heads * (hidden // num_heads))
    k_size = hidden * (num_kv_heads * (hidden // num_heads))
    v_size = hidden * (num_kv_heads * (hidden // num_heads))
    out_size = hidden * hidden

    if "Mixtral" in config["architectures"][0]:
        num_experts = config.get("num_local_experts", 1)
        gate_proj = (intermediate, hidden)
        up_proj = (intermediate, hidden)
        down_proj = (hidden, intermediate)
        ffn_per_layer = num_experts * (gate_proj[0] * gate_proj[1] + up_proj[0] * up_proj[1] + down_proj[0] * down_proj[1])
        router = (num_experts, hidden)
        tensors["ffn_gate_inp"] = router
    else:
        ffn_per_layer = (intermediate, hidden) * 3

    for i in range(layers):
        prefix = f"blk.{i}."
        tensors[prefix + "attn_q.weight"] = (q_size // hidden, hidden) if isinstance(q_size, int) else (num_heads * (hidden // num_heads), hidden)
        tensors[prefix + "attn_k.weight"] = (num_kv_heads * (hidden // num_heads), hidden)
        tensors[prefix + "attn_v.weight"] = (num_kv_heads * (hidden // num_heads), hidden)
        tensors[prefix + "attn_output.weight"] = (hidden, hidden)
        tensors[prefix + "attn_norm.weight"] = (hidden,)
        tensors[prefix + "ffn_norm.weight"] = (hidden,)

        if "Mixtral" in config["architectures"][0]:
            num_experts = config.get("num_local_experts", 1)
            tensors[prefix + "ffn_gate_inp.weight"] = (num_experts, hidden)
            for e in range(num_experts):
                tensors[prefix + f"ffn_gate.{e}.weight"] = (intermediate, hidden)
                tensors[prefix + f"ffn_up.{e}.weight"] = (intermediate, hidden)
                tensors[prefix + f"ffn_down.{e}.weight"] = (hidden, intermediate)
        else:
            tensors[prefix + "ffn_gate.weight"] = (intermediate, hidden)
            tensors[prefix + "ffn_up.weight"] = (intermediate, hidden)
            tensors[prefix + "ffn_down.weight"] = (hidden, intermediate)

    return tensors

def predict_size(config, quant_type="Q4_K_M"):
    tensors = get_tensor_counts(config)
    total_bytes = 0

    bits_map = {
        "F32": 32.0,
        "F16": 16.0,
        "Q8_0": 8.5,
        "Q4_K_M": 4.5
    }
    bpw = bits_map.get(quant_type, 16.0)

    for name, shape in tensors.items():
        nelems = math.prod(shape)
        if len(shape) == 1 or "norm" in name or "inp" in name:
            b = nelems * 4 if quant_type == "F32" and "norm" in name else nelems * 2
        else:
            b = int(nelems * bpw / 8.0)
        total_bytes += b

    return total_bytes
