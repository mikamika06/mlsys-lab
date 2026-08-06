import ref


def check(workdir):
    from collate.adaptive import AdaptiveCollatePlanner

    out = {"adaptive_batching_matched": 0.0, "budget_respected": 0.0}

    sample_gen = ref.make_deterministic_generator(456)
    collate_fn = ref.make_mock_collate_fn(base_cost_ms=2.0, per_item_cost_ms=0.8)

    planner = AdaptiveCollatePlanner(
        collate_fn=collate_fn,
        sample_generator=sample_gen,
        max_budget_ms=10.0
    )

    safe_bs = planner.compute_max_safe_batch_size(max_batch_size=16)
    if safe_bs == 10:
        out["adaptive_batching_matched"] = 1.0

    feature_configs = ref.make_feature_collate_fns()
    opt_plan = planner.optimize_features_and_batch_size(
        candidate_feature_sets=feature_configs,
        target_batch_size=16
    )

    if (opt_plan is not None and
        opt_plan.get("name") == "medium" and
        opt_plan.get("cost_ms", 999.0) <= 10.0):
        out["budget_respected"] = 1.0

    return out
