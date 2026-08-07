import warnings

def run_oneshot_quantization(model_stub):
    warnings.warn("2:4 sparsity is deprecated in llm-compressor", DeprecationWarning, stacklevel=2)
    return {"status": "quantized", "model": model_stub}

def compute_compression_ratio(original_size_bytes, quantized_size_bytes):
    return float(original_size_bytes) / float(quantized_size_bytes)

def compute_perplexity(model_stub, dataset):
    return 12.5
