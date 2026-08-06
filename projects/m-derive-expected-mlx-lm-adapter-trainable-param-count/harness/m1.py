import ref


def check(workdir):
    from mlx_lora_util.derive import expected_adapter_parameters

    ok = 0
    out = {"counts_matched": 0.0}
    for cfg in ref.CONFIGS:
        rank = 8
        targets = ["q_proj", "v_proj"]

        hidden_dim = cfg.get("hidden_size", 4096)
        num_layers = cfg.get("num_hidden_layers", 32)
        num_heads = cfg.get("num_attention_heads", 32)
        num_kv_heads = cfg.get("num_key_value_heads", num_heads)
        head_dim = hidden_dim // num_heads

        expected = 0
        for mod in targets:
            if mod == "q_proj":
                expected += num_layers * (hidden_dim * rank + rank * (num_heads * head_dim))
            elif mod == "v_proj":
                expected += num_layers * (hidden_dim * rank + rank * (num_kv_heads * head_dim))

        got = expected_adapter_parameters(cfg, rank, targets)
        if got == expected:
            ok += 1

    if ok == len(ref.CONFIGS):
        out["counts_matched"] = 1.0
    return out
