import sys

sys.path.insert(0, ".")
from quant.model import create_tiny_model
from quant.engine import get_calibration_data, build_calibration_dataset, quantize_weights
from quant.evaluate import compute_size_ratio, evaluate_error


def test_quantized_size_reduction():
    model = create_tiny_model()
    inputs = get_calibration_data()
    calib = build_calibration_dataset(model, inputs)
    artifact = quantize_weights(model, calib, bits=4)
    ratio = compute_size_ratio(model, artifact)
    assert ratio >= 3.5, f"Expected size ratio >= 3.5, got {ratio}"


def test_quantization_error_bound():
    model = create_tiny_model()
    inputs = get_calibration_data()
    calib = build_calibration_dataset(model, inputs)
    artifact = quantize_weights(model, calib, bits=4)
    assert evaluate_error(model, artifact), "Quantization error exceeds acceptable threshold"
