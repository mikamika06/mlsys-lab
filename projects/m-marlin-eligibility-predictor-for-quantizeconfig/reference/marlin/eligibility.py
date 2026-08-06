def is_marlin_eligible(config):
    if not isinstance(config, dict):
        return False
    bits = config.get("bits")
    if bits not in [4, 8]:
        return False
    if not config.get("sym", True):
        return False
    group_size = config.get("group_size", -1)
    if group_size != -1 and group_size < 32:
        return False
    if config.get("desc_act", False) and bits == 4:
        return False
    return True
