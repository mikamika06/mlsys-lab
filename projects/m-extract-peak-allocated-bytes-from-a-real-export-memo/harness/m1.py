import ref

def check(workdir):
    from memprof.timeline import extract_peak_allocated_bytes
    data = ref.generate_timeline(seed=123)
    want = ref.extract_peak_allocated_bytes(data)
    try:
        got = extract_peak_allocated_bytes(data)
    except Exception as e:
        return {"peak_bytes_rel_err": 1.0, "_note": f"raised exception: {e}"}

    if want == 0:
        err = 0.0 if got == 0 else 1.0
    else:
        err = abs(float(got) - float(want)) / float(want)
    return {"peak_bytes_rel_err": float(err)}
