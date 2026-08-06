import ref


def check(workdir):
    from arena.pte_reader import parse_pte_constants

    out = {"pte_parsed": 0.0, "errors_handled": 0.0}
    valid_ok = 0
    valid_total = 0
    error_ok = 0
    error_total = 0

    for case in ref.PTE_TEST_CASES:
        data = ref.make_pte_data(case["segments"], case["version"], case["magic"])
        if case["expect_valid"]:
            valid_total += 1
            want = ref.reference_parse_pte_constants(data)
            try:
                got = parse_pte_constants(data)
                if got == want:
                    valid_ok += 1
                elif "_note" not in out:
                    out["_note"] = f"mismatch: got {got}, want {want}"
            except Exception as e:
                if "_note" not in out:
                    out["_note"] = f"unexpected error on valid pte: {e}"
        else:
            error_total += 1
            try:
                parse_pte_constants(data)
                if "_note" not in out:
                    out["_note"] = "failed to raise ValueError on invalid pte"
            except ValueError:
                error_ok += 1
            except Exception as e:
                if "_note" not in out:
                    out["_note"] = f"raised wrong exception type: {type(e).__name__}"

    trunc_data = b"PTE1\x01\x00\x00\x00\x05\x00\x00\x00"
    error_total += 1
    try:
        parse_pte_constants(trunc_data)
    except ValueError:
        error_ok += 1

    if valid_total > 0 and valid_ok == valid_total:
        out["pte_parsed"] = 1.0
    if error_total > 0 and error_ok == error_total:
        out["errors_handled"] = 1.0

    return out
