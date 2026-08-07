import sys

def check(workdir):
    sys.path.insert(0, workdir)
    from exporter.model import ModelExporter
    m = {"exported_fully": 0.0}
    exp = ModelExporter({})
    res = exp.export_full()
    if isinstance(res, dict) and res.get("status") == "success":
        m["exported_fully"] = 1.0
    return m
