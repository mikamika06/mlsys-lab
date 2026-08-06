def llama_quantize_mock(tensor_data, quant_type="Q4_0"):
    """Simulate llama-quantize block-wise quantization."""
    raise NotImplementedError


def ollama_requantize_mock(tensor_data, quant_type="Q4_0"):
    """Simulate Ollama-side requantization."""
    raise NotImplementedError


def compare_quantization_drift(tensor_data, quant_type="Q4_0"):
    """Compare quantization MSE and max absolute diff between schemes."""
    raise NotImplementedError
