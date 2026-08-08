import sys
sys.path.insert(0, ".")
import ref

def check(workdir):
    m = {"layers_profiled": 0.0}
    try:
        from int8_eng.profiler import profile_layers
        m1_fp, m1_int = ref.get_mock_models()
        dataset = [1.0, 2.0, 3.0]
        res = profile_layers(m1_fp, m1_int, dataset)
        if isinstance(res, dict) and len(res) == 2:
            m["layers_profiled"] = 1.0
    except Exception:
        pass
    return m
