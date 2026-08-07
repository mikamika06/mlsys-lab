def check(workdir):
    from quant.layout import describe_layout
    m = {"spec_defined": 0.0}
    try:
        info = describe_layout()
        if isinstance(info, dict) and len(info) > 0:
            m["spec_defined"] = 1.0
    except Exception:
        pass
    return m
