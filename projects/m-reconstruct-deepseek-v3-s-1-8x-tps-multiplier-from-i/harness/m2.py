import ref

def check(workdir):
    from mtpcalc.comparison import get_comparison_table
    out = {"table_matched": 0.0}
    try:
        got = get_comparison_table()
        want = ref.get_comparison_table()
        if got == want:
            out["table_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"raised {type(e).__name__}: {str(e)[:100]}"
    return out
