import struct
import sys
sys.path.insert(0, ".")
from tflite_tools.parser import parse_header, extract_op_codes, attribute_bytes
from tflite_tools.stripper import strip_weights


def test_header_identifier():
    buf = bytearray(128)
    buf[0:4] = struct.pack("<I", 16)
    buf[4:8] = b"TFL3"
    h = parse_header(bytes(buf))
    assert h["file_identifier"] == "TFL3"
    assert h["root_table_offset"] == 16


def test_op_codes_extraction():
    buf = bytearray(128)
    root = 16
    struct.pack_into("<I", buf, 0, root)
    struct.pack_into("<I", buf, root, 2)
    struct.pack_into("<I", buf, root + 4, 42)
    struct.pack_into("<I", buf, root + 8, 99)
    codes = extract_op_codes(bytes(buf))
    assert codes == [42, 99]


def test_strip_weights_reduces_or_keeps_valid_structure():
    buf = bytearray(256)
    struct.pack_into("<I", buf, 0, 16)
    struct.pack_into("<I", buf, 20, 500)
    stripped = strip_weights(bytes(buf))
    assert len(stripped) == len(buf)
    attr = attribute_bytes(stripped)
    assert "metadata_bytes" in attr
    assert "buffer_bytes" in attr
