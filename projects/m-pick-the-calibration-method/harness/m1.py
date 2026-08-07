import ref

def check(workdir):
    from calib.picker import pick_calibration_method
    from calib.schema import select_quant_schema

    out = {"methods_matched": 0.0}
    ok = 0

    for i, case in enumerate(ref.TEST_CASES_M1):
        want_calib = ref.pick_calibration_method(case)
        got_calib = pick_calibration_method(case)

        want_schema = ref.select_quant_schema(case["min"], case["max"], case["zp_supp"])
        got_schema = select_quant_schema(case["min"], case["max"], case["zp_supp"])

        if got_calib == want_calib and got_schema == want_schema:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got calib={got_calib}, schema={got_schema}; want calib={want_calib}, schema={want_schema}"

    out["methods_matched"] = float(ok)
    return out
