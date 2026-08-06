from vllm_budget.blocks import predict_num_gpu_blocks
from vllm_budget.kv import bytes_per_token
from vllm_budget.solver import max_context_length


def test_regression_invariants():
    cfg_gqa = {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
    }
    bpt = bytes_per_token(cfg_gqa, "float16")
    assert bpt == 2 * 32 * 8 * 128 * 2

    cfg_explicit_hd = {
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "hidden_size": 3584,
        "head_dim": 256,
    }
    bpt_hd = bytes_per_token(cfg_explicit_hd, "bf16")
    assert bpt_hd == 2 * 28 * 2 * 256 * 2

    max_ctx = max_context_length(cfg_gqa, "fp16", 14_000_000_000, 1_000_000_000, 80_000_000_000)
    assert max_ctx > 0

    blocks = predict_num_gpu_blocks(
        cfg_gqa, "fp16", 80_000_000_000, 0.90, 14_000_000_000, 1_000_000_000, 16
    )
    assert blocks > 0
