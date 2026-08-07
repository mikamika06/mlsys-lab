import ref

def check(workdir):
    from ortprovider.exceptions import safe_create_session, UnregisteredProviderError
    out = {"exceptions_handled": 0.0}
    try:
        safe_create_session(ref.mock_create_fail)
        passed = False
    except UnregisteredProviderError:
        passed = True
    except Exception:
        passed = False

    try:
        res = safe_create_session(ref.mock_create_success)
        success_ok = (res == "session_ok")
    except Exception:
        success_ok = False

    if passed and success_ok:
        out["exceptions_handled"] = 1.0
    else:
        out["_note"] = f"passed={passed}, success_ok={success_ok}"
    return out
