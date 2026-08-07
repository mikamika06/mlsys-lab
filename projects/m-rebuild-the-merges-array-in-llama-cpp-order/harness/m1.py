import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from llama_cpp_tok.merges import rebuild_merges

    out = {"merges_matched": 0.0}
    ok = 0
    total = len(ref.MERGES_TESTS)
    for item in ref.MERGES_TESTS:
        want = sorted(item)
        got = rebuild_merges(item)
        if len(got) == len(want):
            ok += 1
    if ok == total:
        out["merges_matched"] = 1.0
    return out
