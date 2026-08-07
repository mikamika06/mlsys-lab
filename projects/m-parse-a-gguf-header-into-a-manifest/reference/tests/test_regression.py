import sys

sys.path.insert(0, ".")
from gguf_parser.parser import compute_overhead


def test_padding_is_zero_when_aligned():
    manifest = {
        "header_end_offset": 64,
        "_meta_end": 40,
        "metadata": {"general.alignment": 32}
    }
    res = compute_overhead(manifest)
    assert res["padding_waste"] == 0, "Aligned header should have 0 padding waste"


def test_padding_calculated_correctly():
    manifest = {
        "header_end_offset": 60,
        "_meta_end": 40,
        "metadata": {"general.alignment": 32}
    }
    res = compute_overhead(manifest)
    assert res["padding_waste"] == 4, "Padding should be 4 to reach 64"
