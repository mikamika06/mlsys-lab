import ref


def check(workdir):
    from hpabudget.cooldown import compute_safe_cooldown_period
    from hpabudget.readiness import parse_startup_phases

    cases = ref.generate_cooldown_cases()
    total_rel_err = 0.0

    for case in cases:
        phases = parse_startup_phases(case["timestamps"])
        got = compute_safe_cooldown_period(
            phase_times=phases,
            warm_compile_cache=case["warm_cache"],
            cache_speedup_factor=case["speedup"],
            safety_margin_pct=case["margin_pct"],
        )
        expected = ref.ref_compute_cooldown(
            phase_times=ref.ref_parse_startup_phases(case["timestamps"]),
            warm_compile_cache=case["warm_cache"],
            cache_speedup_factor=case["speedup"],
            safety_margin_pct=case["margin_pct"],
        )
        err = abs(got - expected) / float(expected)
        total_rel_err += err

    mean_err = total_rel_err / len(cases)
    out = {"rel_err": mean_err}
    if mean_err > 0.01:
        out["_note"] = f"Relative error in cooldownPeriod calculation too high: {mean_err:.4f}"
    return out
