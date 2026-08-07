import ref
import numpy as np

def check(workdir):
    from exporter.model import business_logic_model
    from exporter.core import analyze_export_stops

    m = {"unsupported_nodes_found": 0.0}
    inp = ref.get_sample_inputs()
    res = analyze_export_stops(business_logic_model, (inp["x"], inp["seq_len"]))
    if isinstance(res, dict) and res.get("unsupported_control_flow"):
        m["unsupported_nodes_found"] = 1.0
    return m
