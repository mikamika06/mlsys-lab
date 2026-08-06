import ref


def check(workdir):
    from picker.payback import build_normalized_table, calculate_payback_volume

    out = {"payback_accurate": 0.0, "table_matched": 0.0}

    test_cases = [
        (30.0, 10.0, 5.0, 6000),
        (60.0, 20.0, 10.0, 6000),
        (10.0, 5.0, 10.0, -1),
        (0.0, 10.0, 5.0, 0),
    ]

    payback_ok = True
    for b_time, b_lat, t_lat, want in test_cases:
        try:
            got = calculate_payback_volume(b_time, b_lat, t_lat)
            if got != want:
                payback_ok = False
                out["_note"] = f"calculate_payback_volume({b_time}, {b_lat}, {t_lat}): expected {want}, got {got}"
                break
        except Exception as e:
            payback_ok = False
            out["_note"] = f"calculate_payback_volume raised {type(e).__name__}: {e}"
            break

    if payback_ok:
        out["payback_accurate"] = 1.0

    try:
        want_table = ref.build_normalized_table(ref.CANDIDATES, baseline_backend="ort_cuda")
        got_table = build_normalized_table(ref.CANDIDATES, baseline_backend="ort_cuda")
        if got_table == want_table:
            out["table_matched"] = 1.0
        elif "_note" not in out:
            out["_note"] = f"table mismatch: got {got_table[:1]}, expected {want_table[:1]}"
    except Exception as e:
        if "_note" not in out:
            out["_note"] = f"build_normalized_table raised {type(e).__name__}: {e}"

    return out
