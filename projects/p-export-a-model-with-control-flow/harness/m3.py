import ref

def check(workdir):
    from exporter.core import declare_dynamic_bounds

    m = {"dynamic_shapes_ok": 0.0}
    spec = declare_dynamic_bounds({"shape": (128,)})
    if isinstance(spec, dict) and "bounds" in spec:
        m["dynamic_shapes_ok"] = 1.0
    return m
