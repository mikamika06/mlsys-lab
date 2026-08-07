def compute_layer_sizes(config):
    bpp = config["bytes_per_param"]
    sizes = []
    for layer in config["layers"]:
        t = layer["type"]
        if t == "embd":
            sizes.append(layer["hidden_dim"] * layer["ffn_dim"] * bpp)
        elif t == "attn":
            h = layer["hidden_dim"]
            kv = layer.get("kv_heads", 1)
            hd = layer.get("head_dim", 128)
            sizes.append((h * h + h * kv * hd * 2) * bpp)
        elif t == "output":
            sizes.append(layer["hidden_dim"] * layer["vocab_size"] * bpp)
        else:
            sizes.append(1024 * bpp)
    return sizes
