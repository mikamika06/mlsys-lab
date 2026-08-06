"""Checker for Milestone 2."""
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from unsloth_bench.parser import parse_unsloth_log, compute_speedup_ratio

    cases = ref.generate_log_cases()
    passed = 0
    for case in cases:
        parsed = parse_unsloth_log(case["log"])
        speedup = compute_speedup_ratio(parsed["steps_per_sec"], case["vanilla_sps"])
        if parsed == case["want"] and abs(speedup - case["want_speedup"]) <= 1e-4:
            passed += 1

    return {"exact_match": 1.0 if passed == len(cases) else 0.0}
