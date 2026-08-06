import sys
sys.path.insert(0, ".")
from miltool.parser import parse_op_histogram
from miltool.builder import build_three_op_program
from miltool.passes import simulate_sdpa_pass

def test_histogram_parsing_accuracy():
    dump = "  transpose : 10\n  matmul : 5\n  softmax : 2\n"
    hist = parse_op_histogram(dump)
    assert hist.get("transpose") == 10
    assert hist.get("matmul") == 5
    assert hist.get("softmax") == 2

def test_three_op_structure():
    prog = build_three_op_program()
    assert len(prog) == 3
    assert prog[0]["name"] == "transpose"
    assert prog[1]["name"] == "matmul"
    assert prog[2]["name"] == "softmax"

def test_sdpa_pass_scaling():
    res = simulate_sdpa_pass(2048)
    assert res["memory_bytes_after"] < res["memory_bytes_before"]
    assert res["ops_after"] < res["ops_before"]
