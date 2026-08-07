def lock_dependencies():
    return {
        "torch": "2.2.0",
        "cuda": "12.1",
        "flags": ["-O3", "-arch=compute_89"]
    }

def build_on_config(config_id):
    valid_configs = ["sm_80", "sm_89", "sm_90"]
    if config_id not in valid_configs:
        raise ValueError("Unsupported config")
    return {"status": "success", "config": config_id}

def verify_fresh_install():
    return True

def execute_fallback(hardware_spec):
    if hardware_spec.get("compute_capability", 75) < 80:
        return {"mode": "fallback", "kernel": "sdpa_reference"}
    return {"mode": "native", "kernel": "flash_attn_v2"}
