import numpy as np


def expected_bits(base_bits: float, lora_bits: float, base_params: int, lora_params: int, double_quant: bool = True) -> float:
    total_params = base_params + lora_params
    dq_bits = 8 / 64 if double_quant else 0.0
    effective_base_bits = base_bits + dq_bits
    total_bits = (base_params * effective_base_bits) + (lora_params * lora_bits)
    return float(total_bits / total_params)
