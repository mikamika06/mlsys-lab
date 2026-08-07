import ref
from qfix.parser import parse_checkpoint
from qfix.metadata import map_metadata

def check(workdir):
    m = {"metadata_ok": 0.0}
    raw, _ = ref.get_sample_data()
    try:
        parsed = parse_checkpoint(raw)
        mapped = map_metadata(parsed)
        if any("model.layers" in k for k in mapped.keys()):
            m["metadata_ok"] = 1.0
    except Exception:
        pass
    return m
