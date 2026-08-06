"""IPEX API Classification module."""

UPSTREAMED_MAP = {
    "ipex.optimize": "torch.xpu.optimize",
    "ipex.cpu.optimize": "torch.compile",
    "ipex.quantization.prepare": "torch.ao.quantization.prepare",
    "ipex.quantization.convert": "torch.ao.quantization.convert",
    "ipex.llm.optimize": "torch.compile",
    "ipex.reflection": "torch.jit.trace",
}

REMOVED_SET = {
    "ipex.core.enable_auto_dnnl",
    "ipex.core.disable_auto_dnnl",
    "ipex.enable_onednn_fusion",
    "ipex.disable_onednn_fusion",
    "ipex.align_dense_memory",
}

RETAINED_SET = {
    "ipex.fast_bert",
    "ipex.nn.functional.interaction",
    "ipex.optimize_transformers",
}


def classify_api_call(api_name):
    """Classify an IPEX API call string into upstreamed, removed, or retained."""
    if api_name in UPSTREAMED_MAP:
        return {"status": "upstreamed", "target": UPSTREAMED_MAP[api_name]}
    if api_name in REMOVED_SET:
        return {"status": "removed", "target": None}
    if api_name in RETAINED_SET:
        return {"status": "retained", "target": api_name}
    return {"status": "unknown", "target": None}


def classify_api_batch(api_list):
    """Classify a list of API name strings."""
    return [classify_api_call(name) for name in api_list]
