def check(workdir):
    from audit.core import extract_kernel_code
    m = {"code_extracted": 0.0}
    try:
        code = extract_kernel_code(None, None)
        if isinstance(code, str) and len(code) > 0:
            m["code_extracted"] = 1.0
    except Exception:
        pass
    return m
