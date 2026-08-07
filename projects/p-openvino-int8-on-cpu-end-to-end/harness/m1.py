import os

def check(workdir):
    m = {"converted_ok": 0.0}
    mod_path = os.path.join(workdir, "ov_engine", "converter.py")
    if not os.path.isfile(mod_path):
        return m

    import sys
    sys.path.insert(0, workdir)
    from ov_engine.converter import convert_model

    out_file = os.path.join(workdir, "test_out.xml")
    try:
        res = convert_model("dummy_source", out_file)
        if res and os.path.exists(out_file):
            m["converted_ok"] = 1.0
    except Exception:
        pass
    finally:
        if os.path.exists(out_file):
            os.remove(out_file)
    return m
