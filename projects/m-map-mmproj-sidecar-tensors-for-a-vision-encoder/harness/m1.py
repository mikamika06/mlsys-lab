import ref

def check(workdir):
    from mmproj.mapping import map_tensors

    raw = [
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.embeddings.class_embedding",
        "vision_model.embeddings.position_embedding.weight",
        "vision_model.post_layernorm.weight",
        "vision_model.post_layernorm.bias",
        "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.linear_1.bias",
        "multi_modal_projector.linear_2.weight",
        "multi_modal_projector.linear_2.bias"
    ]

    want = ref.map_tensors(raw)
    try:
        got = map_tensors(raw)
    except Exception as e:
        return {"exact_match": 0.0, "_note": f"crashed: {e}"}

    if got == want:
        return {"exact_match": 1.0}
    return {"exact_match": 0.0, "_note": f"got {got}, want {want}"}
