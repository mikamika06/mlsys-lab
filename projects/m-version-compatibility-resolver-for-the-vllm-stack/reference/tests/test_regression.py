from vllm_compat.transformers_rewriter import rewrite_v4_to_v5


def test_rewrite_preserves_safetensors_intent():
    snippet = "model = AutoModel.from_pretrained('test')"
    res = rewrite_v4_to_v5(snippet)
    assert "use_safetensors=True" in res, "safetensors default missing in rewritten output"


def test_rewrite_handles_quantization_config():
    snippet = "cfg = AutoConfig.from_pretrained('test', load_in_8bit=True)"
    res = rewrite_v4_to_v5(snippet)
    assert "quantization_config=BitsAndBytesConfig" in res, "bitsandbytes config migration failed"


def test_rewrite_device_placement():
    snippet = "model.half().cuda()"
    res = rewrite_v4_to_v5(snippet)
    assert "model.to(dtype=torch.float16, device='cuda')" in res, "device placement migration failed"
