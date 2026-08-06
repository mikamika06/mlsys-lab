from triton_calc.occupancy import max_resident_blocks


def evaluate_budget(test_cases: list) -> list:
    """Evaluate budget test cases across defined hardware configurations."""
    results = []
    for tc in test_cases:
        blocks = max_resident_blocks(
            tc["regs_per_thread"], tc["threads_per_block"], tc["spec"]
        )
        results.append(
            {
                "regs": tc["regs_per_thread"],
                "threads": tc["threads_per_block"],
                "max_blocks": blocks,
            }
        )
    return results
