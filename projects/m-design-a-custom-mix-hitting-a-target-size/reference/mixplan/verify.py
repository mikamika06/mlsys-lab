def verify_f32_1d(config, recipe):
    for t in config["tensors"]:
        if len(t["shape"]) == 1:
            if recipe.get(t["name"]) != "F32":
                return False
    return True
