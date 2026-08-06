from flopcount.attention import count_attention_flops
from flopcount.transformer import count_layer_flops, count_transformer_flops
from flopcount.varlen import count_varlen_attention_flops, flops_from_histogram

ATTN_TEST_CASES = [
    {"b": 2, "h_q": 32, "h_kv": 8, "s_q": 1024, "s_k": 1024, "d": 128, "causal": False},
    {"b": 4, "h_q": 16, "h_kv": 16, "s_q": 2048, "s_k": 2048, "d": 64, "causal": True},
    {"b": 1, "h_q": 8, "h_kv": 2, "s_q": 512, "s_k": 1024, "d": 128, "causal": False},
    {"b": 2, "h_q": 12, "h_kv": 12, "s_q": 256, "s_k": 512, "d": 64, "causal": True},
]

VARLEN_TEST_CASES = [
    {"seq_lens": [128, 256, 512, 1024], "h_q": 16, "d": 64, "causal": True},
    {"seq_lens": [64, 64, 128, 256, 512], "h_q": 32, "d": 128, "causal": False},
    {"hist": {128: 10, 256: 5, 512: 2}, "h_q": 16, "d": 64, "causal": True},
    {"hist": {64: 20, 1024: 1}, "h_q": 8, "d": 128, "causal": False},
]

TRANSFORMER_TEST_CASES = [
    {
        "num_layers": 32,
        "seq_len": 2048,
        "hidden_dim": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "ffn_hidden_dim": 11008,
        "vocab_size": 32000,
        "causal": True,
        "pass_type": "fwd",
    },
    {
        "num_layers": 24,
        "seq_len": 1024,
        "hidden_dim": 2048,
        "num_heads": 16,
        "num_kv_heads": 16,
        "head_dim": 128,
        "ffn_hidden_dim": 5504,
        "vocab_size": 50257,
        "causal": True,
        "pass_type": "bwd",
    },
    {
        "num_layers": 12,
        "seq_len": 4096,
        "hidden_dim": 768,
        "num_heads": 12,
        "num_kv_heads": 12,
        "head_dim": 64,
        "ffn_hidden_dim": 3072,
        "vocab_size": 30522,
        "causal": False,
        "pass_type": "fwd_bwd",
    },
]
