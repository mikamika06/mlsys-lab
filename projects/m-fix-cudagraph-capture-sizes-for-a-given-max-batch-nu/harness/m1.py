import ref

def check(workdir):
    from specengine.cudagraph import get_capture_sizes
    cases = ref.get_test_cases()
    matched = 0
    for mb, st in cases:
        got = get_capture_sizes(mb, st)
        if isinstance(got, list) and len(got) > 0 and got[-1] == mb:
            matched += 1
    return {"sizes_matched": float(matched)}
