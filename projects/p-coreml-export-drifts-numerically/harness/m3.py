def check(workdir):
    from coreml_export.converter import ModelConverter
    m = {"contract_valid": 0.0}
    conv = ModelConverter()
    if conv.verify_input_contract():
        m["contract_valid"] = 1.0
    return m
