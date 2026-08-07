def rewrite_v4_to_v5(snippet):
    s = snippet
    if "from_pretrained" in s and "use_safetensors" not in s and "quantization_config" not in s:
        s = s.rstrip() + ", use_safetensors=True"
    if "load_in_8bit=True" in s and "quantization_config" not in s:
        s = s.replace("load_in_8bit=True", "quantization_config=BitsAndBytesConfig(load_in_8bit=True)")
    if "model.half().cuda()" in s:
        s = s.replace("model.half().cuda()", "model.to(dtype=torch.float16, device='cuda')")
    return s
