import ref


def check(workdir):
    from residency.memory import measure_wired_limit_effect, verify_zero_copy

    out = {"limit_effect_match": 0.0, "zero_copy_verified": 0.0}
    try:
        ma, mb, limits = ref.LIMIT_TESTS[0]
        res_effect = measure_wired_limit_effect(ma, mb, limits)
        if isinstance(res_effect, list) and len(res_effect) == len(limits):
            out["limit_effect_match"] = 1.0
    except Exception:
        pass

    try:
        res_zc = verify_zero_copy(1024)
        if isinstance(res_zc, dict) and res_zc.get("host_to_device_copies", 1) == 0 and res_zc.get("unified_memory", False) is True:
            out["zero_copy_verified"] = 1.0
    except Exception:
        pass

    return out
