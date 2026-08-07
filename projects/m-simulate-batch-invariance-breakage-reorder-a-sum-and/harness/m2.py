import ref

def check(workdir):
    from batchsim.classify import classify_diffs
    got = classify_diffs(ref.CLASSIFY_CASES)
    want = ref.classify_diffs(ref.CLASSIFY_CASES)
    return {"accuracy": 1.0 if got == want else 0.0}
