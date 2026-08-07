import sys

sys.path.insert(0, ".")
from onnx_reader.reader import estimate_ort_savings, scan_ops, scan_opsets


def test_scan_opsets_parses_wire_format():
    model = b'\x42\x04\x0a\x00\x10\x0c'
    assert scan_opsets(model) == {"": 12}


def test_scan_ops_finds_nested_node():
    model = b'\x3a\x07\x2a\x05\x22\x03Add'
    assert scan_ops(model) == {"Add": 1}


def test_ort_savings_calculates_stripped_bytes():
    model = b'\x3a\x0a\x2a\x08\x1a\x01x\x22\x03Add'
    assert estimate_ort_savings(model) == 3
