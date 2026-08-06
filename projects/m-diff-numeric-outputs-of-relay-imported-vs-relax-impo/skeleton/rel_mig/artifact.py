import numpy as np


def roundtrip_module_test(onnx_bytes: bytes, input_data: np.ndarray) -> bool:
    raise NotImplementedError


def measure_artifact_sizes(onnx_bytes: bytes) -> dict:
    raise NotImplementedError
