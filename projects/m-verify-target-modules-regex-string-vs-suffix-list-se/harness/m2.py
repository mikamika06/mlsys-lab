import ref


def check(workdir):
    from loratarget.matcher import resolve_by_regex, resolve_by_suffix
    from loratarget.stats import compute_param_count, verify_equivalence

    out = {"param_counts_matched": 0.0, "equivalence_checks_matched": 0.0}

    params_ok = True
    equiv_ok = True

    for tree, pattern, suffixes in ref.CONFIGS:
        reg_mods = resolve_by_regex(tree, pattern)
        sfx_mods = resolve_by_suffix(tree, suffixes)

        want_reg_params = ref.compute_param_count(tree, ref.resolve_by_regex(tree, pattern))
        got_reg_params = compute_param_count(tree, reg_mods)

        want_sfx_params = ref.compute_param_count(tree, ref.resolve_by_suffix(tree, suffixes))
        got_sfx_params = compute_param_count(tree, sfx_mods)

        if got_reg_params != want_reg_params or got_sfx_params != want_sfx_params:
            params_ok = False

        want_equiv = ref.verify_equivalence(tree, pattern, suffixes)
        got_equiv = verify_equivalence(tree, pattern, suffixes)

        if want_equiv != got_equiv:
            equiv_ok = False

    mismatched_cfg = (
        ref.TREE_A,
        ".*q_proj$",
        ["proj"],
    )
    want_false = ref.verify_equivalence(*mismatched_cfg)
    got_false = verify_equivalence(*mismatched_cfg)
    if want_false != got_false:
        equiv_ok = False

    out["param_counts_matched"] = 1.0 if params_ok else 0.0
    out["equivalence_checks_matched"] = 1.0 if equiv_ok else 0.0

    return out
