import ref


def check(workdir):
    from hpabudget.readiness import classify_readiness

    cases = ref.generate_test_cases()
    correct = 0
    total = len(cases)

    for case in cases:
        got = classify_readiness(case["logs"], case["http_status"], case["engine_state"])
        if got == case["expected_readiness"]:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    out = {"probe_accuracy": accuracy}
    if accuracy < 1.0:
        out["_note"] = f"Failed {total - correct} readiness probe classifications out of {total}"
    return out
