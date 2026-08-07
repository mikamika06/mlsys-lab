import ref

def check(workdir):
    from checkpoint.index import verify_index
    out = {"indexes_matched": 0.0}
    for idx, fs in ref.generate_index_fixtures():
        got = verify_index(idx, fs)
        want = ref.verify_index(idx, fs)
        if got == want:
            out["indexes_matched"] += 1.0
    return out
