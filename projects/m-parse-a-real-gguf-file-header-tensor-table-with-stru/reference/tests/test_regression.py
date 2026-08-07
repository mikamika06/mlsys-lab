import sys
sys.path.insert(0, ".")
from ggufparser.quant import tensor_byte_size


def test_tensor_byte_size_consistency():
    dims = [512, 512]
    size_f16 = tensor_byte_size(dims, 1)
    assert size_f16 == 512 * 512 * 2
    size_q8 = tensor_byte_size(dims, 8)
    assert size_q8 > 0
    size_q4k = tensor_byte_size(dims, 12)
    assert size_q4k > 0
