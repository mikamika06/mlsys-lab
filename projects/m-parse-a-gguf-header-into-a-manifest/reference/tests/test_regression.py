import sys

sys.path.insert(0, ".")
from gguf_parser.overhead import compute_container_overhead
from harness.ref import GENERATED_FIXTURES


def test_alignment_waste_calculation():
    for fix in GENERATED_FIXTURES:
        res = compute_container_overhead(fix["binary"])
        assert res["alignment_waste"] >= 0
        rem = res["data_offset"] % fix["alignment"]
        assert rem == 0
        assert res["header_padding"] == res["data_offset"] - res["header_size"]
        assert res["total_overhead"] == res["data_offset"] + res["alignment_waste"]
