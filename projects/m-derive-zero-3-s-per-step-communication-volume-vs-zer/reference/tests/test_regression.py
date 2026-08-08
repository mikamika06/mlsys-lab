import sys
sys.path.insert(0, ".")
from zerocheck.analysis import compute_communication_volumes
from zerocheck.parser import parse_memory_reduction
from zerocheck.partition import extract_partition_sizes

def test_communication_volume_scaling():
    v = compute_communication_volumes(1000000, 4, 8)
    assert v["zero3"] > v["zero1"]

def test_parser_accuracy():
    lines = ["MEM_REPORT rank=0 z1=1000 z3=400"]
    res = parse_memory_reduction(lines)
    assert res[0]["zero1"] == 1000

def test_partition_sum():
    lines = ["PARAM_INIT name=layer.weight numel=10"]
    res = extract_partition_sizes(lines, 4)
    assert sum(res["layer.weight"]) == 10
