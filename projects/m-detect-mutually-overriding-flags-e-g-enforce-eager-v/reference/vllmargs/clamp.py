def find_clamping_arg(requested_tokens, config):
    limit = config.get("max_model_len", 4096) * config.get("max_num_seqs", 256)
    if requested_tokens > limit:
        return "max_num_seqs or max_model_len"
    if "block_size" in config and requested_tokens % config["block_size"] != 0:
        return "block_size alignment"
    return None
