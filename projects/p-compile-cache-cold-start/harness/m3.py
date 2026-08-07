def check(workdir):
    import os
    import tempfile
    from compcache.engine import CompilerEngine
    m = {"portable_cache": 0.0}
    eng1 = CompilerEngine()
    eng1.compile_and_run(99)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        eng1.export_cache(tmp_path)
        eng2 = CompilerEngine()
        eng2.import_cache(tmp_path)
        res, comps = eng2.compile_and_run(99)
        if comps == 0 and res == 198:
            m["portable_cache"] = 1.0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return m
