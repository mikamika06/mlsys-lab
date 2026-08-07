import ref

def check(workdir):
    from graphbreak.constructs import classify_constructs
    items = [x[0] for x in ref.CONSTRUCT_TESTS]
    want = ref.classify_constructs(items)
    try:
        got = classify_constructs(items)
    except Exception as e:
        return {"constructs_matched": 0.0, "_note": f"raised {type(e).__name__}"}
    match = 0.0
    if isinstance(got, list) and len(got) == len(want):
        for w, g in zip(want, got):
            if w == g:
                match += 1.0
    return {"constructs_matched": match}
