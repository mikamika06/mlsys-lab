COMPAT_TABLE = {
    "525.60.13": ["2022.4", "2023.1"],
    "535.54.03": ["2023.2", "2023.3"],
}

def check_compat(driver, ncu):
    supported = COMPAT_TABLE.get(driver, [])
    return ncu in supported
