import sys
sys.path.insert(0, ".")
from ggufsize.calculator import compute_size
from ggufsize.config import parse_config

def test_tied_embeddings_effect():
    cfg_tied = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 32000,
        "tie_word_embeddings": True
    }
    cfg_untied = dict(cfg_tied)
    cfg_untied["tie_word_embeddings"] = False

    size_tied = compute_size(cfg_tied, "F16")
    size_untied = compute_size(cfg_untied, "F16")
    assert size_untied > size_tied, "Untied embeddings should cost more bytes than tied embeddings"

def test_moe_scaling():
    cfg_dense = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 32000
    }
    cfg_moe = {
        "architectures": ["MixtralForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 32000,
        "num_local_experts": 8,
        "num_experts_per_tok": 2
    }
    size_dense = compute_size(cfg_dense, "F16")
    size_moe = compute_size(cfg_moe, "F16")
    assert size_moe > size_dense * 3, "MoE model with multiple experts must be significantly larger than dense counterpart"

def test_tensor_parsing_non_empty():
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 32000
    }
    tensors = parse_config(cfg)
    assert len(tensors) > 0, "Parsed tensors dictionary cannot be empty"
