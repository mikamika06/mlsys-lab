def check_fp8_availability(config):
    head_dim = config.get("head_dim", 128)
    sm_version = config.get("sm_version", 80)
    has_oca = config.get("has_oca", True)
    if sm_version >= 90 and head_dim % 16 == 0 and has_oca:
        return True
    return False
