def compute_vision_layout(mapped_tensors: dict, config: dict) -> dict:
    patch_weight = mapped_tensors.get("vision.patch_embed.weight")
    if patch_weight is None:
        raise ValueError("Missing patch embedding tensor")

    out_ch, in_ch, p_h, p_w = patch_weight["shape"]
    image_size = config.get("image_size", 336)
    num_patches = (image_size // p_h) * (image_size // p_w)

    proj_w0 = mapped_tensors.get("projector.0.weight")
    if proj_w0 is None:
        raise ValueError("Missing projector initial projection tensor")

    proj_in = proj_w0["shape"][1]
    if proj_in != out_ch:
        raise ValueError(f"Vision dim mismatch: {out_ch} vs projector in {proj_in}")

    proj_out = proj_w0["shape"][0]
    proj_w2 = mapped_tensors.get("projector.2.weight")
    if proj_w2 is not None:
        proj_out = proj_w2["shape"][0]

    return {
        "num_patches": num_patches,
        "vision_dim": out_ch,
        "text_dim": proj_out,
        "patch_size": p_h,
        "in_channels": in_ch,
    }


def calculate_projection_bytes(mapped_tensors: dict) -> int:
    total_bytes = 0
    dtype_sizes = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}

    for name, tensor in mapped_tensors.items():
        if not (name.startswith("vision.") or name.startswith("projector.")):
            continue
        num_elements = 1
        for dim in tensor["shape"]:
            num_elements *= dim
        bpe = dtype_sizes.get(tensor.get("dtype", "float32"), 4)
        total_bytes += num_elements * bpe

    return total_bytes
