import sys
sys.path.insert(0, ".")
from onnxdecode.parser import decode_onnx_graph
from onnxdecode.initializers import dump_initializers
from onnxdecode.splitter import split_weights_and_graph


def test_decode_basic_structure():
    dummy = b"\x12\x06\x0a\x04test"
    res = decode_onnx_graph(dummy)
    assert isinstance(res, dict)
    assert "name" in res


def test_dump_initializers_count():
    graph = {"initializers": [{"name": "w1", "dims": [2, 2], "data_type": 1, "raw_data": b"\x00"*16}]}
    m = dump_initializers(graph)
    assert len(m) == 1
    assert m[0]["size_bytes"] == 16


def test_split_weights_exact():
    data = b"A" * 100 + b"B" * 100
    g, w = split_weights_and_graph(data)
    assert len(g) == 100
    assert len(w) == 100
