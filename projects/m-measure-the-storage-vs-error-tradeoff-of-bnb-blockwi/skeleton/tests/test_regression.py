import sys

sys.path.insert(0, ".")
from bnbquant.quantize import blockwise_quantize, blockwise_dequantize
from bnbquant.evaluate import compute_storage_bytes, compute_mse

def test_quantization_roundtrip_error():
    raise NotImplementedError

def test_storage_calculation_positive():
    raise NotImplementedError
