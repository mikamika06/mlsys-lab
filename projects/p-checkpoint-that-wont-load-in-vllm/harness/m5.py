import ref
from qfix.engine import load_in_engine

def check(workdir):
    m = {"engine_ok": 0.0}
    raw, inputs = ref.get_sample_data()
    try:
        out = load_in_engine(raw, inputs)
        if isinstance(out, (int, float)):
            m["engine_ok"] = 1.0
    except Exception:
        pass
    return m
