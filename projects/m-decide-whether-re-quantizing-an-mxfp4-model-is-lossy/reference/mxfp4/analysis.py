def is_requantization_lossy(config):
    source_format = config.get("source_format", "mxfp4")
    target_format = config.get("target_format", "mxfp4")
    block_size = config.get("block_size", 32)
    if source_format == "mxfp4" and target_format == "mxfp4":
        return True
    if block_size != config.get("target_block_size", block_size):
        return True
    return True
