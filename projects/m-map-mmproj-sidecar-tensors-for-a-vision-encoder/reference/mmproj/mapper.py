import re

PREFIX_PATTERNS = [
    (r"^(v|vision_tower|mm_vision_model)\.patch_embd\.", "vision.patch_embed."),
    (r"^(v|vision_tower|mm_vision_model)\.class_embd", "vision.class_embed"),
    (r"^(v|vision_tower|mm_vision_model)\.position_embd", "vision.position_embed"),
    (r"^(v|vision_tower|mm_vision_model)\.ln_post\.", "vision.ln_post."),
    (r"^(v|vision_tower|mm_vision_model)\.blk\.(\d+)\.", r"vision.blk.\2."),
    (r"^(v|vision_tower|mm_vision_model)\.layers\.(\d+)\.", r"vision.blk.\2."),
    (r"^(mm|mmproj|multi_modal_projector)\.0\.", "projector.0."),
    (r"^(mm|mmproj|multi_modal_projector)\.2\.", "projector.2."),
    (r"^(mm|mmproj|multi_modal_projector)\.l0\.", "projector.0."),
    (r"^(mm|mmproj|multi_modal_projector)\.l2\.", "projector.2."),
]

SUFFIX_PATTERNS = [
    (r"\.attn\.qkv\.", ".attn.qkv."),
    (r"\.attn\.out\.", ".attn.out."),
    (r"\.mlp\.c_fc\.", ".mlp.fc1."),
    (r"\.mlp\.c_proj\.", ".mlp.fc2."),
    (r"\.ln_1\.", ".ln_1."),
    (r"\.ln_2\.", ".ln_2."),
]


def map_sidecar_tensor_name(raw_name: str) -> str:
    mapped = raw_name
    for pat, repl in PREFIX_PATTERNS:
        if re.search(pat, mapped):
            mapped = re.sub(pat, repl, mapped)
            break

    for pat, repl in SUFFIX_PATTERNS:
        mapped = re.sub(pat, repl, mapped)

    if mapped == raw_name and not (mapped.startswith("vision.") or mapped.startswith("projector.")):
        raise KeyError(f"Unrecognized sidecar tensor key: {raw_name}")

    return mapped


def map_sidecar_tensors(raw_dict: dict) -> dict:
    return {map_sidecar_tensor_name(k): v for k, v in raw_dict.items()}
