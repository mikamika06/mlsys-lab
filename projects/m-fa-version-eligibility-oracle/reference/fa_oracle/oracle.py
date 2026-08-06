import packaging.version

def check_eligibility(stack_info):
    cc_str = stack_info.get("compute_capability", "0.0")
    cc = tuple(map(int, cc_str.split(".")))
    fa = stack_info.get("flash_attn_version")
    fa_ver = packaging.version.Version(fa) if fa else None

    eligible = {}
    eligible["fa1"] = cc >= (7, 0) and fa_ver is not None and fa_ver.major == 1
    eligible["fa2"] = cc >= (8, 0) and fa_ver is not None and fa_ver.major == 2 and fa_ver.minor >= 5
    eligible["fa3"] = cc >= (9, 0) and fa_ver is not None and fa_ver.major >= 2 and fa_ver.minor >= 6
    return eligible
