def resolve_recipe(tensor_name, base_ftype):
    if "attn_q" in tensor_name or "ffn_gate" in tensor_name:
        return "Q4_K_M"
    return base_ftype
