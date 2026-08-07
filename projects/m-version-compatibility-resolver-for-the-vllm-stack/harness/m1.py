import ref


def check(workdir):
    from vllm_compat.resolver import check_compatibility

    matched = 0
    total = len(ref.VERSION_TESTS)
    for t in ref.VERSION_TESTS:
        got = check_compatibility(t["stack"], t["constraints"])
        want = ref.resolve_compatibility(t["stack"], t["constraints"])
        if got == want:
            matched += 1

    return {"versions_matched": float(matched), "total": float(total)}
