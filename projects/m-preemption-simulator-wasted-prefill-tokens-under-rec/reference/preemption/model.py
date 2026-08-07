def find_breakeven_point(kv_bytes_per_token, transfer_bandwidth, recompute_cost_per_token):
    if recompute_cost_per_token <= 0:
        return 0
    return int(transfer_bandwidth / recompute_cost_per_token * kv_bytes_per_token)
