TRANSCRIPTS = [
    ("Error: The application returned an error code: ERR_NVGPUCTRPERM", "nvgpuctrperm"),
    ("Failed to load driver symbols, version mismatch between driver and ncu", "driver_mismatch"),
    ("Profiling session timed out waiting for kernel completion", "timeout"),
    ("Out of memory on device while allocating counter buffer", "out_of_memory"),
]

def disambiguate(transcript):
    if "ERR_NVGPUCTRPERM" in transcript:
        return "nvgpuctrperm"
    if "version mismatch" in transcript:
        return "driver_mismatch"
    if "timed out" in transcript:
        return "timeout"
    return "out_of_memory"

PERM_TESTS = [
    {"regkey": 0, "groups": ["users"], "root": False, "expect": True},
    {"regkey": 1, "groups": ["nsight"], "root": False, "expect": False},
    {"regkey": 0, "groups": [], "root": True, "expect": False},
    {"regkey": 2, "groups": ["users"], "root": False, "expect": True},
]

def predict_perm(regkey, groups, root):
    if root:
        return False
    if regkey == 1 and "nsight" in groups:
        return False
    return True

COMPAT_TABLE = {
    "525.60.13": ["2022.4", "2023.1"],
    "535.54.03": ["2023.2", "2023.3"],
}

def check_compat(driver, ncu):
    supported = COMPAT_TABLE.get(driver, [])
    return ncu in supported
