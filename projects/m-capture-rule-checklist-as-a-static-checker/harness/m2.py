import ref

def check(workdir):
    from graph_checker.checker import check_graph_violations
    from graph_checker.optimizer import suggest_safe_transforms

    cases = ref.generate_test_cases()
    transforms_valid = 1.0
    clears_violations = 1.0

    for gm, expected_count in cases:
        if expected_count > 0:
            try:
                opt_gm = suggest_safe_transforms(gm)
                remaining = check_graph_violations(opt_gm)
                if len(remaining) != 0:
                    clears_violations = 0.0
            except Exception:
                transforms_valid = 0.0
                clears_violations = 0.0

    return {
        "transforms_valid": transforms_valid,
        "clears_violations": clears_violations
    }
