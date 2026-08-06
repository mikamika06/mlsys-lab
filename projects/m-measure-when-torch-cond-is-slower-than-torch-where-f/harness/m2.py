import ref


def check(workdir):
    from condcheck.decision import decide_branch_strategy
    out = {"cases_correct": 0.0}
    correct = 0
    for case in ref.IR_CASES:
        got = decide_branch_strategy(case)
        if got == case["expected"]:
            correct += 1
    out["cases_correct"] = float(correct)
    return out
