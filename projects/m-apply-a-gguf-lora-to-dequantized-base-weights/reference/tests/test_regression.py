import numpy as np
from adapter.convert import convert_peft_to_gguf
from gguf_adapter.parser import parse_lora_gguf_and_build_delta
from gguf_adapter.dequant import apply_lora_to_dequantized_base


def test_lora_application_scaling():
    r = 4
    alpha = 16.0
    scaling = alpha / r
    peft_dict = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight.default": np.ones((r, 8), dtype=np.float32),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight.default": np.ones((16, r), dtype=np.float32)
    }
    gguf_dict = convert_peft_to_gguf(peft_dict, alpha)
    target = "layers.0.self_attn.q_proj"
    parsed = parse_lora_gguf_and_build_delta(gguf_dict, target)

    expected_delta = scaling * (peft_dict["base_model.model.layers.0.self_attn.q_proj.lora_B.weight.default"] @
                                peft_dict["base_model.model.layers.0.self_attn.q_proj.lora_A.weight.default"])
    np.testing.assert_allclose(parsed["delta"], expected_delta, rtol=1e-5, atol=1e-5)

    base_weights = {target: np.zeros((16, 8), dtype=np.float32)}
    fused = apply_lora_to_dequantized_base(base_weights, {target: parsed})
    np.testing.assert_allclose(fused[target], expected_delta, rtol=1e-5, atol=1e-5)
