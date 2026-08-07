def parse_overrides(args_list):
    overrides = {}
    for arg in args_list:
        if "=" in arg:
            k, v = arg.split("=", 1)
            overrides[k.strip()] = v.strip().upper()
    return overrides
