import numpy as np
from qfix.parser import parse_checkpoint
from qfix.metadata import map_metadata
from qfix.packing import fix_packing
from qfix.verify import verify_output

def load_in_engine(raw_data, inputs):
    parsed = parse_checkpoint(raw_data)
    meta = map_metadata(parsed)
    packed = fix_packing(meta)
    return verify_output(packed, inputs)
