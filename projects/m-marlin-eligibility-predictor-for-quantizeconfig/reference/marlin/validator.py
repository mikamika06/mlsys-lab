class QuantizationConfigError(ValueError):
    pass

def validate_quantize_config(config):
    if not isinstance(config, dict):
        raise QuantizationConfigError("Config must be a dictionary")
    if "bits" not in config:
        raise QuantizationConfigError("Missing 'bits' field")
    bits = config["bits"]
    if bits not in [2, 3, 4, 8]:
        raise QuantizationConfigError(f"Unsupported bits: {bits}")
    group_size = config.get("group_size", -1)
    if group_size != -1 and group_size <= 0:
        raise QuantizationConfigError(f"Invalid group_size: {group_size}")
    if not isinstance(config.get("sym", True), bool):
        raise QuantizationConfigError("Field 'sym' must be a boolean")
    return True
