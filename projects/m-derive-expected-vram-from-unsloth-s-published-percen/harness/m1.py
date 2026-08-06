"""Checker for Milestone 1."""
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from unsloth_bench.vram import vram_expected_gb

    cases = ref.generate_vram_cases()
    passed = 0
    for case in cases:
        got = vram_expected_gb(case["vanilla"], case["pct"])
        if abs(got - case["expected"]) <= 1e-4:
            passed += 1

    return {"exact_match": 1.0 if passed == len(cases) else 0.0}
