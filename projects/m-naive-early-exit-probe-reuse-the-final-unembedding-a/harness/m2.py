import ref


def check(workdir):
    from earlyexit.probe import sweep_and_compare
    _, w, f, hidden_dict, published_table = ref.generate_fixture()
    want_res = ref.sweep_and_compare(hidden_dict, w, f, published_table)
    try:
        got_res = sweep_and_compare(hidden_dict, w, f, published_table)
    except Exception as e:
        return {"curve_err": 1.0, "_note": f"raised {type(e).__name__}: {e}"}

    got_diff = got_res.get("mean_diff", 1.0)
    want_diff = want_res.get("mean_diff", 0.0)
    curve_err = float(abs(got_diff - want_diff))
    return {"curve_err": curve_err}
