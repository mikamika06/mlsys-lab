import ref

def check(workdir):
    out = {"compute_matched": 0.0, "load_matched": 0.0}
    try:
        from memplan.predict import compute_buffer_bytes, load_time_seconds
    except ImportError:
        out["_note"] = "could not import memplan.predict"
        return out
    
    ok_compute = 0
    for args in ref.FIXTURES_PREDICT:
        want = ref.compute_buffer_bytes(*args)
        try:
            got = compute_buffer_bytes(*args)
            if got == want:
                ok_compute += 1
        except Exception:
            pass
    if ref.FIXTURES_PREDICT:
        out["compute_matched"] = float(ok_compute) / len(ref.FIXTURES_PREDICT)

    ok_load = 0
    for args in ref.FIXTURES_LOAD:
        want = ref.load_time_seconds(*args)
        try:
            got = load_time_seconds(*args)
            if abs(got - want) < 1e-6:
                ok_load += 1
        except Exception:
            pass
    if ref.FIXTURES_LOAD:
        out["load_matched"] = float(ok_load) / len(ref.FIXTURES_LOAD)

    return out
