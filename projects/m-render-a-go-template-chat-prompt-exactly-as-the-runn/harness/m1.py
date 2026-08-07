import ref


def check(workdir):
    from promptfmt.template import render
    total = len(ref.RENDER_CASES)
    matched = 0
    for case in ref.RENDER_CASES:
        got = render(case["template"], case["system"], case["messages"])
        if got == case["expected"]:
            matched += 1
    frac = float(matched) / float(total) if total > 0 else 0.0
    return {"byte_exact_fraction": frac}
