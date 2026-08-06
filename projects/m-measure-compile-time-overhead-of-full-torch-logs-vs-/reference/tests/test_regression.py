import sys

sys.path.insert(0, ".")
from tlog.unsupported import extract_unsupported_op
from tlog.overhead import compute_overhead_ratio


def test_extract_unsupported_op_valid():
    sample = "}[TORCH_LOGS] dynamo: [WARNING] graph break due to unsupported op: aten.foo.default"
    assert extract_unsupported_op(sample) == "aten.foo.default"


def test_compute_overhead_ratio_positive():
    assert compute_overhead_ratio(10.0, 12.0) == 1.2
