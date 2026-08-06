import ref

def check(workdir):
    from graph_checker.checker import check_graph_violations

    cases = ref.generate_test_cases()
    matched = 0
    total = len(cases)
    
    for gm, expected_count in cases:
        violations = check_graph_violations(gm)
        if len(violations) == expected_count:
            matched += 1

    return {
        "violations_matched": 1.0 if matched == total else 0.0
    }
