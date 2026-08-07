import ref


def check(workdir):
    from kvparse.capacity import analyze_tp_scaling
    out = {"scaling_correct": 0.0}
    ok = 0
    for l1, l2, expected_double in ref.TP_LOGS:
        res = analyze_tp_scaling(l1, l2)
        if res["doubles_capacity"] == expected_double:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"scaling test failed for logs: got {res}"
    if ok == len(ref.TP_LOGS):
        out["scaling_correct"] = 1.0
    return out
