def check(workdir):
    from coreml_export.converter import ModelConverter
    m = {"conversion_fixed": 0.0}
    conv = ModelConverter()
    if conv.fix_problematic_ops():
        m["conversion_fixed"] = 1.0
    return m
