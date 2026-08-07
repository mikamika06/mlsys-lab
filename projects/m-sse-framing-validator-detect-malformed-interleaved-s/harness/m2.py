import ref

def check(workdir):
    from ssevall.mapper import map_openai_request_to_sampling_params

    out = {"params_matched": 0.0}
    ok = 0
    for req, expected in ref.REQUESTS:
        try:
            got = map_openai_request_to_sampling_params(req)
            if got == expected:
                ok += 1
        except Exception:
            pass
    out["params_matched"] = float(ok)
    return out
