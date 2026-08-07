def sweep_tp_performance(config, tp_degrees):
    raise NotImplementedError


def max_valid_tp_degree(num_kv_heads, num_attention_heads):
    raise NotImplementedError


def verify_server_log_partitions(log_lines, expected_tp):
    raise NotImplementedError
