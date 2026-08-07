def validate_device_target(device_str, available_devices):
    if not isinstance(device_str, str) or not device_str.strip():
        raise RuntimeError("Cannot open empty device name string.")
    target = device_str.split(":")[0].strip()
    if target not in available_devices:
        raise RuntimeError(f"Device '{device_str}' is not in list of supported devices: {available_devices}")
    return True
