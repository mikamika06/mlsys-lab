import ref


def check(workdir):
    from mixplan.verify import verify_f32_1d

    out = {"invariants_verified": 0.0, "caught_violations": 0.0}
    verified_ok = 0
    caught_ok = 0

    for cfg in ref.CONFIGS:
        good_recipe = ref.solve_recipe(cfg, cfg["budget_bytes"])
        if verify_f32_1d(cfg, good_recipe):
            verified_ok += 1

        bad_recipe = dict(good_recipe)
        for t in cfg["tensors"]:
            if len(t["shape"]) == 1:
                bad_recipe[t["name"]] = "Q8_0"
                break

        if not verify_f32_1d(cfg, bad_recipe):
            caught_ok += 1

    out["invariants_verified"] = 1.0 if verified_ok == len(ref.CONFIGS) else 0.0
    out["caught_violations"] = 1.0 if caught_ok == len(ref.CONFIGS) else 0.0
    return out
