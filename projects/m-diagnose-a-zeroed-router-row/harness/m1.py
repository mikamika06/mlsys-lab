import ref


def check(workdir):
    from moediag.router import find_zeroed_rows

    out = {"rows_diagnosed": 0.0}
    weights = ref.generate_router_weights()
    got = find_zeroed_rows(weights)
    if isinstance(got, list) and 3 in got:
        out["rows_diagnosed"] = 1.0
    else:
        out["_note"] = f"expected row 3 to be zeroed, got {got}"
    return out
