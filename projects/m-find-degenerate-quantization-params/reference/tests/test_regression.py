import sys
sys.path.insert(0, ".")
import struct
from tflite_tools.quant import find_degenerate_quantization_params
from tflite_tools.diff import structural_diff
from tflite_tools.rebuild import rebuild_flatbuffer

def test_find_degenerate():
    data = bytearray(b"TFL3")
    data.extend(struct.pack("<I", 2) + b"t1" + struct.pack("<fI", 0.0, 0))
    res = find_degenerate_quantization_params(bytes(data))
    assert res == [0]

def test_structural_diff():
    b1 = bytearray(b"TFL3")
    b1.extend(struct.pack("<I", 2) + b"t1" + struct.pack("<fI", 0.1, 0))
    b2 = bytearray(b"TFL3")
    b2.extend(struct.pack("<I", 2) + b"t2" + struct.pack("<fI", 0.1, 0))
    diff = structural_diff(bytes(b1), bytes(b2))
    assert len(diff["changed_tensors"]) == 1

def test_rebuild_flatbuffer():
    b1 = bytearray(b"TFL3")
    b1.extend(struct.pack("<I", 2) + b"t1" + struct.pack("<fI", 0.1, 0))
    out = rebuild_flatbuffer(bytes(b1), {"t1": "t_updated"})
    assert b"t_updated" in out
