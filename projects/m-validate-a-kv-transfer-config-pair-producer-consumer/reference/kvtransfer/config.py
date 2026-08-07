def validate_pair(producer_cfg: dict, consumer_cfg: dict) -> dict:
    """Validate producer/consumer KV transfer configuration pair."""
    errors = []

    if producer_cfg.get("role") != "producer":
        errors.append("producer role must be 'producer'")
    if consumer_cfg.get("role") != "consumer":
        errors.append("consumer role must be 'consumer'")

    if producer_cfg.get("num_layers") != consumer_cfg.get("num_layers"):
        errors.append("mismatched num_layers")

    if producer_cfg.get("num_kv_heads") != consumer_cfg.get("num_kv_heads"):
        errors.append("mismatched num_kv_heads")

    if producer_cfg.get("head_dim") != consumer_cfg.get("head_dim"):
        errors.append("mismatched head_dim")

    if producer_cfg.get("block_size") != consumer_cfg.get("block_size"):
        errors.append("mismatched block_size")

    if producer_cfg.get("dtype") != consumer_cfg.get("dtype"):
        errors.append("mismatched dtype")

    prod_transport = producer_cfg.get("transport", {})
    cons_transport = consumer_cfg.get("transport", {})

    if prod_transport.get("type") != cons_transport.get("type"):
        errors.append("mismatched transport type")

    if prod_transport.get("type") == "rdma":
        if prod_transport.get("gid") == cons_transport.get("gid") and prod_transport.get("qp_num") == cons_transport.get("qp_num"):
            errors.append("duplicate RDMA QP endpoint")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }
