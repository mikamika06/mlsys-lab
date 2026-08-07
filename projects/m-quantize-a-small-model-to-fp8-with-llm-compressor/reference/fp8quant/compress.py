def simulate_compression(recipe):
    if recipe.get("quant_method") == "fp8":
        return 500, 1000
    return 1000, 1000
