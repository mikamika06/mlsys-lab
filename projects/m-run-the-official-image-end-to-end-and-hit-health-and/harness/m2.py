import ref


def check(workdir):
    from vllm_runner.client import parse_health, parse_completion

    out = {"health_matched": 0.0, "completion_matched": 0.0}

    h_test_cases = [
        (200, "OK"),
        (500, "ERROR"),
        (200, ""),
    ]
    h_ok = True
    for status, body in h_test_cases:
        want = ref.parse_health(status, body)
        try:
            got = parse_health(status, body)
        except Exception:
            got = None
        if got != want:
            h_ok = False
            out["_note"] = f"health parse mismatch for status={status}, body={body}: got {got}, want {want}"
            break
    if h_ok:
        out["health_matched"] = 1.0

    c_test_cases = [
        {"choices": [{"text": "hello world"}]},
        {"choices": [{"text": ""}]},
        {},
    ]
    c_ok = True
    for payload in c_test_cases:
        try:
            want = ref.parse_completion(payload)
        except Exception as e:
            want = e

        try:
            got = parse_completion(payload)
        except Exception as e:
            got = e

        if isinstance(want, Exception):
            if not isinstance(got, type(want)):
                c_ok = False
                out["_note"] = f"completion parse exception mismatch for {payload}: got {type(got)}, want {type(want)}"
                break
        else:
            if got != want:
                c_ok = False
                out["_note"] = f"completion parse mismatch for {payload}: got {got}, want {want}"
                break
    if c_ok:
        out["completion_matched"] = 1.0

    return out
