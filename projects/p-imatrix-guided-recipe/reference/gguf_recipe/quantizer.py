from gguf_recipe.selector import select_quant_types

def build_recipes(weights, imatrix):
    r_with = select_quant_types(weights, imatrix)
    r_without = {name: "Q4_0" for name in weights}
    return r_with, r_without
