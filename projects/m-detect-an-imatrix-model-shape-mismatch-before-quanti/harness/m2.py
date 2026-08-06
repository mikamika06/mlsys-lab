import ref

def check(workdir):
    from imatrix.validator import validate_imatrix
    tg, ig, tb, ib = ref.get_test_cases()
    res_good = validate_imatrix(tg, ig)
    res_bad = validate_imatrix(tb, ib)
    match = 1.0 if (res_good["valid"] and not res_bad["valid"]) else 0.0
    return {"report_matches": match}
