def find_reusable_blocks(config):
    total = config["total_tokens"]
    bs = config["block_size"]
    num_blocks = total // bs
    reusable = []
    for b_idx in range(num_blocks):
        block_end = (b_idx + 1) * bs
        is_reusable = True
        for layer in config["layers"]:
            if layer["kind"] == "sliding":
                window = layer["window"]
                if block_end <= (total - window):
                    is_reusable = False
                    break
        if is_reusable:
            reusable.append(b_idx)
    return reusable
