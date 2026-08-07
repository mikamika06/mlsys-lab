import ref


def check(workdir):
    from graphclean.parser import parse_graph_breaks

    out = {"parsed_correctly": 0.0}
    try:
        res = parse_graph_breaks(ref.LOG_DATA)
        want = ref.PARSED_RESULT
        if res == want:
            out["parsed_correctly"] = 1.0
        else:
            out["_note"] = f"got {res}, want {want}"
    except Exception as e:
        out["_note"] = f"raised {type(e).__name__}: {str(e)[:120]}"
    return out
