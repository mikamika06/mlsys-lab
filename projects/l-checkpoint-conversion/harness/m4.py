import ref


def check(workdir):
    from convertkit import names, shapes

    out = {"shape_reversed": 0.0, "attention_ok": 0.0, "gqa_detected": 0.0,
           "head_dim_from_metadata": 0.0, "matches_mlx_layer": 0.0}

    doc = ref.gguf_index("llama")
    t = next(x for x in doc["tensors"] if x["name"] == "blk.0.attn_q.weight")
    if shapes.target_shape(t["shape_ggml_order"]) == list(
            reversed(t["shape_ggml_order"])):
        out["shape_reversed"] = 1.0

    got = shapes.check_attention(doc["tensors"], doc["metadata"], "llama")
    if not got.get("problems"):
        out["attention_ok"] = 1.0
    want = got.get("expected", {})
    if want.get("grouped") is True and want.get("group_size") == 4:
        out["gqa_detected"] = 1.0
    # 5120 // 32 is 160; this checkpoint uses 128 and says so in the metadata.
    if want.get("head_dim") == 128 and want.get("q_out") == 4096:
        out["head_dim_from_metadata"] = 1.0

    mapping = names.map_index(doc["tensors"])
    mlx = {k for k in ref.mlx_params() if k.startswith("layers.0.")}
    if names.layer_targets(mapping, 0) == mlx:
        out["matches_mlx_layer"] = 1.0
    return out
