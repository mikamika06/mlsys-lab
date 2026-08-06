import sys
sys.path.insert(0, ".")
from ggufsize.calc import model_total_bytes, tensor_bytes
from ggufsize.model import is_output_tensor, parse_tensors


def test_output_tensor_exclusion():
    tensors = [
        {"name": "token_embd.weight", "shape": [100, 100], "ftype": 0},
        {"name": "output.weight", "shape": [100, 100], "ftype": 0}
    ]
    size_with = model_total_bytes(tensors, 0, leave_output=True)
    size_without = model_total_bytes(tensors, 0, leave_output=False)
    assert size_with > size_without
    assert size_with - size_without == 40000


def test_tensor_byte_computation():
    shape = [32, 32]
    b = tensor_bytes(shape, 0)
    assert b == 4096
