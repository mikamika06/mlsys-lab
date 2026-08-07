def define_state_contract(num_layers, num_heads, head_dim, max_seq_len):
    contract = {
        "inputs": ["input_ids", "cache_seq_len"],
        "states": [],
        "outputs": ["logits"]
    }
    for i in range(num_layers):
        k_name = f"key_cache_{i}"
        v_name = f"val_cache_{i}"
        shape = (1, num_heads, max_seq_len, head_dim)
        contract["states"].append({"name": k_name, "shape": shape})
        contract["states"].append({"name": v_name, "shape": shape})
    return contract
