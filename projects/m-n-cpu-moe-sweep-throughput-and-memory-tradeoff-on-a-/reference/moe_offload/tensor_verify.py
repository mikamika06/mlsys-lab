def verify_override_tensor(tensor_names, overrides):
    mapping = {}
    for name in tensor_names:
        device = "gpu"
        for pat, dev in overrides.items():
            if pat in name:
                device = dev
        mapping[name] = device
    return mapping
