import ref


def check(workdir):
    from profiler.diagnose import diagnose_cache_behavior

    cases = ref.generate_workload_cases()
    correct = 0
    for i, c in enumerate(cases):
        res = diagnose_cache_behavior(
            c["l2_capacity_bytes"],
            c["working_set_bytes"],
            c["measured_l2_hit_rate"]
        )
        expected_working_set_large = c["working_set_bytes"] > c["l2_capacity_bytes"]
        if res.get("is_working_set_large") == expected_working_set_large:
            correct += 1

    out = {"diagnosis_match": 1.0 if correct == len(cases) else 0.0}
    if correct < len(cases):
        out["_note"] = f"Passed {correct}/{len(cases)} diagnostic cases"
    return out
