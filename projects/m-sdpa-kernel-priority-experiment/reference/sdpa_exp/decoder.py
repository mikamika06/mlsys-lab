def can_use_flash_attention(config):
    if config["head_dim"] not in (32, 64, 128, 256):
        return False
    if config["dtype_bytes"] == 4 and config["seq_len"] > 2048:
        return False
    return True
