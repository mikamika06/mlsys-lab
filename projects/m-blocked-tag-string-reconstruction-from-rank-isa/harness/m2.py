import ref


def check(workdir):
    from dnnfmt.logs import count_reorders

    ok = 1
    for i, (text, want) in enumerate(ref.LOGS):
        got = count_reorders(text)
        if got != want:
            ok = 0
            break
    return {"counts_matched": float(ok)}
