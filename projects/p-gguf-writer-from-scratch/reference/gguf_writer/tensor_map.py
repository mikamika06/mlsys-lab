def map_tensor_name(name):
    mapping = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.layers.0.self_attn.q_proj.weight": "blk.0.attn_q.weight",
        "model.layers.0.self_attn.k_proj.weight": "blk.0.attn_k.weight",
        "model.layers.0.self_attn.v_proj.weight": "blk.0.attn_v.weight",
        "model.layers.0.self_attn.o_proj.weight": "blk.0.attn_output.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight"
    }
    return mapping.get(name, name)
