def hybrid_usp_bandwidth(config, nvlink_bw, infiniband_bw):
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    return float((nvlink_bw * kv_heads + infiniband_bw * heads) / (nvlink_bw + infiniband_bw + 1e-5))
