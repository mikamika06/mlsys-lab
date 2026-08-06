import ref


def check(workdir):
    from collate.budget import profile_collate, evaluate_budget

    out = {"profile_matched": 0.0, "budget_eval_matched": 0.0}

    sample_gen = ref.make_deterministic_generator(123)
    collate_fn = ref.make_mock_collate_fn(base_cost_ms=1.0, per_item_cost_ms=0.5)
    batch_sizes = [4, 8, 16, 32]

    profile_got = profile_collate(collate_fn, sample_gen, batch_sizes)

    profile_ok = True
    for bs in batch_sizes:
        if bs not in profile_got:
            profile_ok = False
            break
        expected_time = 1.0 + 0.5 * bs
        got_time = profile_got[bs]["total_time_ms"]
        if abs(got_time - expected_time) > 1e-4:
            profile_ok = False
            break

    if profile_ok:
        out["profile_matched"] = 1.0

    eval_got = evaluate_budget(
        profile_results=profile_got,
        max_budget_ms_per_batch=10.0,
        target_throughput_samples_sec=1000.0
    )

    eval_ok = True
    for bs in batch_sizes:
        if bs not in eval_got:
            eval_ok = False
            break
        expected_time = 1.0 + 0.5 * bs
        expected_within = expected_time <= 10.0
        expected_tp = (bs / (expected_time / 1000.0))
        expected_tp_ok = expected_tp >= 1000.0

        got_item = eval_got[bs]
        if (got_item["within_budget"] != expected_within or
            got_item["meets_throughput"] != expected_tp_ok or
            got_item["compliant"] != (expected_within and expected_tp_ok)):
            eval_ok = False
            break

    if eval_ok:
        out["budget_eval_matched"] = 1.0

    return out
