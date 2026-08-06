def probe_head_dim_ceiling(hardware_spec):
    sm = hardware_spec.get("sm", 90)
    if sm >= 90:
        return {"fa2_max_head_dim": 256, "fa3_max_head_dim": 128}
    return {"fa2_max_head_dim": 128, "fa3_max_head_dim": 0}

def compare_throughput(version, head_dim, seq_len):
    base = seq_len * 0.001
    if version == "fa3":
        if head_dim <= 128:
            return 300.0 + base
        return 150.0 + base
    else:
        if head_dim <= 256:
            return 220.0 + base
        return 100.0 + base

def check_fp8_availability(head_dim, sm_version):
    if sm_version < 90:
        return False, "requires Hopper sm_90 or higher"
    if head_dim % 16 != 0:
        return False, "head_dim must be a multiple of 16"
    if head_dim > 128:
        return False, "FP8 attention limited to head_dim <= 128 on current hardware kernels"
    return True, "available"
