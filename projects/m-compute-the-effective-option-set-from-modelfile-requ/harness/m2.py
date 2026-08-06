import ref

def check(workdir):
    from runneropts.split import classify_options
    traces = {
        "num_ctx": {"load_duration_changed": True},
        "num_gpu": {"load_duration_changed": True},
        "temperature": {"load_duration_changed": False},
        "top_p": {"load_duration_changed": False}
    }
    try:
        lt, st = classify_options(traces)
        expected_lt = ["num_ctx", "num_gpu"]
        expected_st = ["temperature", "top_p"]
        if sorted(lt) == sorted(expected_lt) and sorted(st) == sorted(expected_st):
            return {"split_matched": 1.0}
    except Exception:
        pass
    return {"split_matched": 0.0}
