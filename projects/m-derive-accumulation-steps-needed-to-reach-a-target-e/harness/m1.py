import ref

def check(workdir):
    from accum.derivation import compute_accumulation_steps
    cases = ref.get_test_cases()
    passed = 0
    for case in cases:
        got = compute_accumulation_steps(
            case["target_effective_batch_size"],
            case["per_device_batch_size"],
            case["num_devices"]
        )
        if got == case["expected_steps"]:
            passed += 1

    match = 1.0 if passed == len(cases) else 0.0
    return {"steps_matched": match}
