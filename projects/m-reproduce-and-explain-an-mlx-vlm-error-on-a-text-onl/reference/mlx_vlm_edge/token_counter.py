def compute_image_tokens(resolution, patch_size, vision_config=None):
    h, w = resolution if isinstance(resolution, tuple) else (resolution, resolution)
    grid_h = h // patch_size
    grid_w = w // patch_size
    tokens = grid_h * grid_w
    if vision_config and "spatial_merge_size" in vision_config:
        merge = vision_config["spatial_merge_size"]
        tokens = (grid_h // merge) * (grid_w // merge)
    return tokens
