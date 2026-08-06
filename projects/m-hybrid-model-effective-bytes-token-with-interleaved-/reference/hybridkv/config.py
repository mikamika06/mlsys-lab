def classify_attention(layer_cfg):
    if layer_cfg.get("window") is not None and layer_cfg.get("window") > 0:
        return "sliding"
    return "full"
