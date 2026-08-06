import torch
from qlora.derive import expected_bits
from qlora.quant import quantize_and_measure
from qlora.compare import compare_nf4_fp4


def get_test_tensor():
    torch.manual_seed(1337)
    return torch.randn(32, 32)
