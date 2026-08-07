def will_fallback_to_cpu(quant_type: str, backend_config: dict) -> bool:
    if not isinstance(quant_type, str) or not quant_type.startswith("IQ"):
        return False
    name = backend_config.get("name", "cpu")
    if name == "cpu":
        return False
    supported = set(backend_config.get("supported_iq_types", []))
    if quant_type in supported:
        return False
    arch = backend_config.get("arch_version", 0.0)
    if name == "cuda" and arch >= 8.0:
        return False
    if name == "metal" and arch >= 3.0:
        return False
    return True
