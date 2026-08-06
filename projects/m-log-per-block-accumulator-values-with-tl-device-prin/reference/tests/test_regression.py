import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from debugger.logger import parse_device_print
from debugger.masking import extract_program_id

def test_parse_multi_digit_coordinates():
    stdout = "[10, 20, 0] acc_val: 42.0\n[0, 100, 5] acc_val: -3.14"
    res = parse_device_print(stdout)
    assert (10, 20, 0) in res
    assert res[(10, 20, 0)] == 42.0
    assert (0, 100, 5) in res
    assert res[(0, 100, 5)] == -3.14

def test_extract_program_id_multi_digit():
    err = "ValueError: out of bounds memory access at program_id (12, 34, 56)"
    assert extract_program_id(err) == (12, 34, 56)

def test_extract_program_id_no_match():
    err = "RuntimeError: segmentation fault"
    assert extract_program_id(err) == ()
