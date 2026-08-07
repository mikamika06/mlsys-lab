import ref

def check(workdir):
    m = {"tactics_match": 0.0}
    try:
        from trt_builder.inspector import verify_tactics
        ea, eb = ref.get_sample_engines()
        if verify_tactics(ea, eb) is True:
            m["tactics_match"] = 1.0
    except Exception:
        pass
    return m
