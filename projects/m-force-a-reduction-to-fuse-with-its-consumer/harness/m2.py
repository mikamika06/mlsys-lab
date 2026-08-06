import ref


def check(workdir):
    from fusion.analyzer import count_vectorized_loops

    out = {"vector_loops_matched": 0.0}
    try:
        got = count_vectorized_loops(ref.CPP_DUMP)
        want = ref.count_vectorized_loops(ref.CPP_DUMP)
        if got == want:
            out["vector_loops_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = str(e)
    return out
