def check_compatibility(config, runtime):
    bits = config.get("bits", 4)
    desc_act = config.get("desc_act", False)
    group_size = config.get("group_size", 128)

    if runtime == "exllamav2":
        if bits not in [2, 3, 4, 8]:
            return False
        if group_size not in [-1, 128]:
            return False
        return True
    elif runtime == "autogptq":
        return True
    elif runtime == "gptqmodel":
        return True
    elif runtime == "triton":
        if desc_act and bits == 2:
            return False
        return True
    else:
        return False
