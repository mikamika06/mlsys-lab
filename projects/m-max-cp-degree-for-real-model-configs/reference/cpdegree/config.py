def max_cp_degree(config):
    return min(config["num_key_value_heads"], config["num_attention_heads"])
