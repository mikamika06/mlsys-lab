import ref


def check(workdir):
    from mixplan.solver import solve_recipe
    from mixplan.recipe import recipe_bytes

    out = {"size_ratio": 0.0, "valid_recipes": 0.0}
    max_ratio = 0.0
    valid_count = 0

    for i, cfg in enumerate(ref.CONFIGS):
        budget = cfg["budget_bytes"]
        rec = solve_recipe(cfg, budget)
        if not isinstance(rec, dict):
            out["_note"] = f"config {i}: solver returned non-dict"
            return out

        cost = recipe_bytes(cfg, rec)
        ratio = cost / float(budget)
        if ratio > max_ratio:
            max_ratio = ratio

        ref_rec = ref.solve_recipe(cfg, budget)
        ref_cost = ref.recipe_bytes(cfg, ref_rec)

        if cost <= budget and cost >= ref_cost * 0.9:
            valid_count += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: cost {cost} vs budget {budget} (ref cost {ref_cost})"

    out["size_ratio"] = float(max_ratio)
    out["valid_recipes"] = 1.0 if valid_count == len(ref.CONFIGS) else 0.0
    return out
