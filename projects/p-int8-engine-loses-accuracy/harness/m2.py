import sys
sys.path.insert(0, ".")
import ref

def check(workdir):
    m = {"sensitivity_identified": 0.0}
    try:
        from int8_eng.profiler import profile_layers
        from int8_eng.tuning import find_sensitive_layers
        m1_fp, m1_int = ref.get_mock_models()
        prof = profile_layers(m1_fp, m1_int, [1.0, 2.0])
        sens = find_sensitive_layers(prof, threshold=0.05)
        if "sensitive_layer" in sens and "safe_layer" not in sens:
            m["sensitivity_identified"] = 1.0
    except Exception:
        pass
    return m
