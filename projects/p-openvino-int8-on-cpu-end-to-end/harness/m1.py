import os
import ref

def check(workdir):
    m = {"converted_ok": 0.0}
    out_path = os.path.join(workdir, "test_model.bin")
    try:
        from cpuopt.converter import convert_model
        res = convert_model("dummy", out_path)
        if os.path.exists(out_path) and res.get("status") == "success":
            m["converted_ok"] = 1.0
    except Exception:
        pass
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
    return m
