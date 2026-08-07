import sys

sys.path.insert(0, ".")
from tritonop.chain import reconstruct_chain
from tritonop.parser import parse_kernel


def test_parse_non_empty():
    code = "x = tl.load(ptr)\ny = x + 1"
    ops = parse_kernel(code)
    assert len(ops) == 2


def test_reconstruct_basic():
    code = "x = tl.load(ptr)\ny = x + 1"
    chain = reconstruct_chain(code)
    assert "aten.load" in chain
    assert "aten.add" in chain


def test_chain_order():
    code = "a = tl.load(p1)\nb = a * 2"
    chain = reconstruct_chain(code)
    assert chain == ["aten.load", "aten.mul"]
