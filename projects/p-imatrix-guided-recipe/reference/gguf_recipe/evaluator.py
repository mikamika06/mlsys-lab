import numpy as np

def evaluate_recipe(weights, recipe):
    ppl_vals = []
    kld_vals = []
    for name in weights:
        t = recipe.get(name, "Q4_0")
        base_ppl = 8.0 if t == "Q4_0" else 4.5
        ppl_vals.append(base_ppl)
        kld_vals.append(0.15 if t == "Q4_0" else 0.04)
    return {"ppl": float(np.mean(ppl_vals)), "kld": float(np.mean(kld_vals))}

def measure_gain(weights, recipe_with, recipe_without):
    m_with = evaluate_recipe(weights, recipe_with)
    m_without = evaluate_recipe(weights, recipe_without)
    return m_without["ppl"] - m_with["ppl"]
