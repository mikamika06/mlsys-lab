def predict_nvgpuctrperm(rm_profiling_admin_only, is_root, user_groups):
    val = int(rm_profiling_admin_only)
    if val == 0:
        return {"will_fail": False, "reason": "ALLOWED_ALL_USERS"}

    if is_root:
        return {"will_fail": False, "reason": "ALLOWED_ROOT"}

    groups = [g.lower() for g in user_groups]
    if val == 1:
        return {"will_fail": True, "reason": "ERR_NVGPUCTRPERM_ADMIN_ONLY"}

    if val == 2:
        if "video" in groups or "nvgpubase" in groups or "tracer" in groups:
            return {"will_fail": False, "reason": "ALLOWED_GROUP_MEMBER"}
        return {"will_fail": True, "reason": "ERR_NVGPUCTRPERM_GROUP_RESTRICTED"}

    return {"will_fail": True, "reason": "ERR_NVGPUCTRPERM_UNKNOWN_RESTRICTION"}
