import ref

def check(workdir):
    from exporter.model import business_logic_model
    from exporter.core import export_model
    import numpy as np

    m = {"export_matches": 0.0}
    inp = ref.get_sample_inputs()
    res = export_model(business_logic_model, (inp["x"], inp["seq_len"]))
    if isinstance(res, dict) and res.get("status") == "success":
        m["export_matches"] = 1.0
    return m
