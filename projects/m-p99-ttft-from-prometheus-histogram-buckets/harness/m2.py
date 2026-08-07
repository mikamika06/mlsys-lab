import ref


def check(workdir):
    from kvobs.alerting import HysteresisAlert
    from kvobs.triage import triage_system_state

    out = {"triage_matched": 0.0, "alert_matched": 0.0}

    triage_ok = True
    for metrics, base_ttft, want in ref.TRIAGE_TEST_CASES:
        try:
            got = triage_system_state(metrics, base_ttft)
            if got != want:
                triage_ok = False
                out["_note"] = f"triage got '{got}', want '{want}' for {metrics}"
                break
        except Exception as e:
            triage_ok = False
            out["_note"] = f"triage raised exception: {e}"
            break

    if triage_ok:
        out["triage_matched"] = 1.0

    alert_ok = True
    for case in ref.ALERT_TEST_CASES:
        alert = HysteresisAlert(case["high"], case["low"], case["hold"])
        got_outputs = []
        want_outputs = case["expected"]
        for v in case["inputs"]:
            try:
                got_outputs.append(alert.process(v))
            except Exception as e:
                alert_ok = False
                out["_note"] = f"alert process raised exception: {e}"
                break
        if got_outputs != want_outputs:
            alert_ok = False
            if "_note" not in out:
                out["_note"] = f"alert history got {got_outputs}, want {want_outputs}"
            break

    if alert_ok:
        out["alert_matched"] = 1.0

    return out
