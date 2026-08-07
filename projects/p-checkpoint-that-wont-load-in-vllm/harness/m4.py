import ref
from qfix.parser import parse_checkpoint
from qfix.metadata import map_metadata
from qfix.packing import fix_packing
from qfix.verify import verify_output

def check(workdir):
    m = {"output_match": 0.0}
    raw, inputs = ref.get_sample_data()
    try:
        parsed = parse_checkpoint(raw)
        mapped = map_metadata(parsed)
        packed = fix_packing(mapped)
        out = verify_output(packed, inputs)
        if isinstance(out, (int, float)) and out != 0.0:
            m["output_match"] = 1.0
    except Exception:
        pass
    return m
