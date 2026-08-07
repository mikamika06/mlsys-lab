import sys
sys.path.insert(0, ".")
from edge.pte import build_pte

def test_pte_header_magic():
    data = build_pte("test_payload")
    assert data[:4] == b"\x50\x54\x45\x31", "Invalid PTE magic header"

def test_pte_header_length():
    payload = "hello_world"
    data = build_pte(payload)
    import struct
    length = struct.unpack("<I", data[4:8])[0]
    assert length == len(payload), "PTE header length mismatch"
