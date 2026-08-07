VERSION_TESTS = [
    {
        "stack": {"vllm": "0.4.0", "torch": "2.2.0", "bitsandbytes": "0.42.0", "transformers": "4.38.0"},
        "constraints": {"min_vllm": "0.3.0", "max_vllm": "0.5.0", "min_torch": "2.1.0"},
        "expected": True
    },
    {
        "stack": {"vllm": "0.2.0", "torch": "2.0.0", "bitsandbytes": "0.41.0", "transformers": "4.30.0"},
        "constraints": {"min_vllm": "0.3.0", "max_vllm": "0.5.0", "min_torch": "2.1.0"},
        "expected": False
    },
    {
        "stack": {"vllm": "0.4.1", "torch": "2.3.0", "bitsandbytes": "0.43.0", "transformers": "4.40.0"},
        "constraints": {"min_vllm": "0.4.0", "max_vllm": "0.5.0", "min_torch": "2.2.0"},
        "expected": True
    }
]

BNB_TESTS = [
    {
        "env": {"HAS_CUDA": True, "BNB_VERSION": "0.43.0", "COMPILE_FLAG": 1},
        "expected": {"backend": "cuda", "features": ["int8", "fp4", "nf4", "optim"], "supported": True}
    },
    {
        "env": {"HAS_CUDA": False, "BNB_VERSION": "0.41.0", "COMPILE_FLAG": 0},
        "expected": {"backend": "cpu", "features": ["int8"], "supported": False}
    },
    {
        "env": {"HAS_CUDA": True, "BNB_VERSION": "0.39.0", "COMPILE_FLAG": 1},
        "expected": {"backend": "cuda", "features": ["int8"], "supported": True}
    }
]

REWRITE_TESTS = [
    {
        "snippet": "from transformers import PreTrainedModel\nmodel = PreTrainedModel.from_pretrained('foo')",
        "expected": "from transformers import PreTrainedModel\nmodel = PreTrainedModel.from_pretrained('foo', use_safetensors=True)"
    },
    {
        "snippet": "config = AutoConfig.from_pretrained('bar', load_in_8bit=True)",
        "expected": "config = AutoConfig.from_pretrained('bar', quantization_config=BitsAndBytesConfig(load_in_8bit=True))"
    },
    {
        "snippet": "model.half().cuda()",
        "expected": "model.to(dtype=torch.float16, device='cuda')"
    }
]


def resolve_compatibility(stack, constraints):
    def parse(v):
        return tuple(int(x) for x in v.split("."))

    vllm_v = parse(stack["vllm"])
    torch_v = parse(stack["torch"])

    if "min_vllm" in constraints and vllm_v < parse(constraints["min_vllm"]):
        return False
    if "max_vllm" in constraints and vllm_v > parse(constraints["max_vllm"]):
        return False
    if "min_torch" in constraints and torch_v < parse(constraints["min_torch"]):
        return False
    return True


def detect_bnb(env):
    has_cuda = env.get("HAS_CUDA", False)
    ver = env.get("BNB_VERSION", "0.0.0")
    major, minor, _ = (int(x) for x in ver.split("."))

    backend = "cuda" if has_cuda else "cpu"
    features = ["int8"]
    if major > 0 or (major == 0 and minor >= 40):
        if has_cuda:
            features.extend(["fp4", "nf4"])
    if major > 0 or (major == 0 and minor >= 42):
        features.append("optim")

    supported = has_cuda and (major > 0 or (major == 0 and minor >= 40))
    return {"backend": backend, "features": sorted(features), "supported": supported}


def rewrite_snippet(snippet):
    import re
    s = snippet
    if "from_pretrained" in s and "use_safetensors" not in s and "quantization_config" not in s:
        s = s.rstrip() + ", use_safetensors=True"
    if "load_in_8bit=True" in s and "quantization_config" not in s:
        s = s.replace("load_in_8bit=True", "quantization_config=BitsAndBytesConfig(load_in_8bit=True)")
    if "model.half().cuda()" in s:
        s = s.replace("model.half().cuda()", "model.to(dtype=torch.float16, device='cuda')")
    return s
