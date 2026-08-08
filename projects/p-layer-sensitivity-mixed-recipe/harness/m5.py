def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import quant
    import ref
    model, x = ref.get_fixture()
    budget = 3000
    out = quant.compare_recipes(model, x, budget, [8, 4, 2])
    mixed_wins = 1.0 if out["mixed_mse"] < out["uniform_mse"] else 0.0
    return {"mixed_wins": mixed_wins}
