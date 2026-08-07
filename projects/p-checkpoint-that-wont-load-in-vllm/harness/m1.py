import ref
from qfix.parser import parse_checkpoint

def check(workdir):
    m = {"format_ok": 0.0}
    raw, _ = ref.get_sample_data()
    try:
        res = parse_checkpoint(raw)
        if isinstance(res, dict) and len(res) > 0:
            m["format_ok"] = 1.0
    except Exception:
        pass
    return m
