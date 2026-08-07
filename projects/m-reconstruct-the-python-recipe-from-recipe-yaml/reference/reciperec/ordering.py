def validate_ordering(recipe_dict):
    order_map = {
        "constant_folding_modifier": 0,
        "sparsity_modifier": 1,
        "smooth_quant_modifier": 2,
        "quantization_modifier": 3,
    }
    for stage in recipe_dict.get("stages", []):
        for stage_name, mods in stage.items():
            last_idx = -1
            for mod_name in mods.keys():
                idx = order_map.get(mod_name, 99)
                if idx < last_idx:
                    return False
                last_idx = idx
    return True
