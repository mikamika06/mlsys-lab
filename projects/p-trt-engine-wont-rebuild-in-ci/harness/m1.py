import ref

def check(workdir):
    m = {"compatibility_check_ok": 0.0}
    try:
        from trt_builder.compatibility import check_compatibility
        meta = ref.get_sample_metadata()
        if check_compatibility(meta) is True:
            m["compatibility_check_ok"] = 1.0
    except Exception:
        pass
    return m
