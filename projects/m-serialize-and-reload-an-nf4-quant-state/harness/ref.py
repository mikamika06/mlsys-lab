import torch

class MockQuantState:
    def __init__(self, absmax, bits, quant_type):
        self.absmax = torch.tensor(absmax, dtype=torch.float32)
        self.bits = bits
        self.quant_type = quant_type

CONFIGS = [
    MockQuantState([0.1, 0.5, -0.3], 4, "nf4"),
    MockQuantState([1.0, 2.0, 3.0], 4, "nf4"),
    MockQuantState([-0.5, 0.0, 0.5], 8, "fp8")
]

METADATA_SAMPLES = [
    {"bits": "4", "bnb_4bit_quant_type": "nf4", "quant_method": "bitsandbytes"},
    {"bits": "4", "bnb_4bit_quant_type": "fp4", "quant_method": "bitsandbytes", "bnb_4bit_use_double_quant": "true"},
    {"quantization_config": '{"bits": 4, "bnb_4bit_quant_type": "nf4"}'}
]
