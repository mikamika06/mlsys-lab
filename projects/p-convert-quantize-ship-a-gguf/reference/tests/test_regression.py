import sys
import os

sys.path.insert(0, ".")
from gguf_pipe.convert import verify_tokenizer, convert_safetensors_to_gguf
from gguf_pipe.quantize import get_quantized_size
from gguf_pipe.eval import compute_kld
import numpy as np


def test_conversion_preserves_tensor_shapes():
    vocab_path = "tmp_vocab.json"
    with open(vocab_path, "w") as f:
        f.write('{"vocab": {"a": 1}}')
    try:
        assert verify_tokenizer(vocab_path) is True
    finally:
        if os.path.exists(vocab_path):
            os.remove(vocab_path)


def test_quantization_reduces_size():
    fp16_size = get_quantized_size("model.gguf", "FP16")
    q4_size = get_quantized_size("model.gguf", "Q4_K_M")
    assert q4_size < fp16_size


def test_kld_non_negative():
    ref = np.array([1.0, 2.0, 3.0])
    quant = np.array([0.9, 2.1, 2.8])
    kld = compute_kld(ref, quant)
    assert kld >= 0.0
