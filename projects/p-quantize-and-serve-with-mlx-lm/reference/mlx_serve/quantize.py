import ref

def quantize_model(bits: int, max_gb: float) -> dict:
    return ref.simulate_quantization(bits, max_gb)
