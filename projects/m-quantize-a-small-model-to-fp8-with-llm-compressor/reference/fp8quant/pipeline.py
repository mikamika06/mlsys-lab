from fp8quant.recipe import build_recipe
from fp8quant.compress import simulate_compression


def run_compression():
    recipe = build_recipe()
    comp_size, orig_size = simulate_compression(recipe)
    return float(comp_size) / float(orig_size)
