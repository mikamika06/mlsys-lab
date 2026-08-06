from flopcount.transformer import count_layer_flops, count_transformer_flops


def test_transformer_flops_swiglu():
    flops = count_layer_flops(
        seq_len=1024,
        hidden_dim=4096,
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        ffn_hidden_dim=11008,
        causal=True,
        pass_type="fwd",
    )
    assert flops > 0


def test_transformer_pass_type_multiplier():
    fwd = count_layer_flops(
        seq_len=512,
        hidden_dim=1024,
        num_heads=8,
        num_kv_heads=8,
        head_dim=128,
        ffn_hidden_dim=4096,
        causal=True,
        pass_type="fwd",
    )
    bwd = count_layer_flops(
        seq_len=512,
        hidden_dim=1024,
        num_heads=8,
        num_kv_heads=8,
        head_dim=128,
        ffn_hidden_dim=4096,
        causal=True,
        pass_type="bwd",
    )
    fwd_bwd = count_layer_flops(
        seq_len=512,
        hidden_dim=1024,
        num_heads=8,
        num_kv_heads=8,
        head_dim=128,
        ffn_hidden_dim=4096,
        causal=True,
        pass_type="fwd_bwd",
    )
    assert bwd == 2 * fwd
    assert fwd_bwd == 3 * fwd
