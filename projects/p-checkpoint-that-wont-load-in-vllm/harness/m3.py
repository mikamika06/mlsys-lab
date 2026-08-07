import ref
from qfix.parser import parse_checkpoint
from qfix.metadata import map_metadata
from qfix.packing import fix_packing

def check(workdir):
    m = {"packing_ok": 0.0}
    raw, _ = ref.get_sample_data()
    try:
        parsed = parse_checkpoint(raw)
        mapped = map_metadata(parsed)
        packed = fix_packing(mapped)
        if len(packed) > 0:
            m["packing_ok"] = 1.0
    except Exception:
        pass
    return m
