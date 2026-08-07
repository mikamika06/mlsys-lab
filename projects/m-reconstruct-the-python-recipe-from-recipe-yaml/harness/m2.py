import ref


def check(workdir):
    from reciperec.ordering import validate_ordering
    from reciperec.modules import count_modules

    out = {"ordering_matched": 0.0, "counts_matched": 0.0}

    ordering_ok = True
    bad_recipe = {
        "version": "1.0",
        "stages": [{
            "s": {
                "quantization_modifier": {"bits": 4},
                "sparsity_modifier": {"target_sparsity": 0.5}
            }
        }]
    }
    for cfg in ref.RECIPES:
        if validate_ordering(cfg) != ref.validate_ordering(cfg):
            ordering_ok = False
            break
    if validate_ordering(bad_recipe) is True:
        ordering_ok = False

    if ordering_ok:
        out["ordering_matched"] = 1.0

    counts_ok = True
    for cfg in ref.RECIPES:
        want_cnt = ref.count_modules(cfg, ref.MODEL_MODULES)
        try:
            got_cnt = count_modules(cfg, ref.MODEL_MODULES)
        except Exception:
            counts_ok = False
            break
        if got_cnt != want_cnt:
            counts_ok = False
            break

    if counts_ok:
        out["counts_matched"] = 1.0

    return out
