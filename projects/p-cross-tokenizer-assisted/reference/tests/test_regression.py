import sys
sys.path.insert(0, ".")
from speculative.alignment import align_bytes
from speculative.transfer import transfer_candidates

def test_unknown_chars():
    vocab = [b"a", b"b"]
    res = align_bytes([99, 97], vocab)
    assert isinstance(res, list)
    mapped = transfer_candidates([99], {97: 1})
    assert mapped == [0]
