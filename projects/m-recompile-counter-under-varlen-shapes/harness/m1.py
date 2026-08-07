import ref


def check(workdir):
    from recompile.counter import count_recompiles

    out = {"recompile_count_match": 0.0}
    want = ref.count_recompiles(ref.SHAPES_FIXTURE)
    try:
        got = count_recompiles(ref.SHAPES_FIXTURE)
        if got == want:
            out["recompile_count_match"] = 1.0
        else:
            out["_note"] = f"got recompile count {got}, want {want}"
    except Exception as e:
        out["_note"] = f"raised {type(e).__name__}: {e}"
    return out
