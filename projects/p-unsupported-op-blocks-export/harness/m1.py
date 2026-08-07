import sys

def check(workdir):
    sys.path.insert(0, workdir)
    from exporter.model import ModelExporter
    m = {"localized": 0.0}
    exp = ModelExporter({})
    op = exp.localize_unsupported()
    if op == "CustomGelu":
        m["localized"] = 1.0
    return m
