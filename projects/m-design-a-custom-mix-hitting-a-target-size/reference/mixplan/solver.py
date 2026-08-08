from mixplan.recipe import recipe_bytes


def solve_recipe(config, budget_bytes):
    recipe = {}
    for t in config["tensors"]:
        if len(t["shape"]) == 1:
            recipe[t["name"]] = "F32"

    matrix_tensors = [t for t in config["tensors"] if len(t["shape"]) > 1]
    sorted_matrices = sorted(matrix_tensors, key=lambda x: x.get("importance", 0.5), reverse=True)

    for t in sorted_matrices:
        recipe[t["name"]] = "Q4_K"

    candidates = ["F32", "F16", "Q8_0", "Q4_K"]

    for t in sorted_matrices:
        for ftype in candidates:
            recipe[t["name"]] = ftype
            if recipe_bytes(config, recipe) <= budget_bytes:
                break
        else:
            recipe[t["name"]] = "Q4_K"

    return recipe
