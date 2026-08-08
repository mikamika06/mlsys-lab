from plan_compat.parser import MAGIC


def classify_deserialization(header: dict, runtime: dict) -> str:
    if not header.get("valid_checksum") or header.get("magic") != MAGIC:
        return "CORRUPTED_HEADER"

    h_trt = header["trt_version"]
    r_trt = runtime["trt_version"]

    if h_trt[0] != r_trt[0]:
        return "TRT_VERSION_MISMATCH"
    if h_trt > r_trt:
        return "TRT_VERSION_MISMATCH"
    if (
        r_trt > h_trt
        and not header["hardware_compatible"]
        and h_trt[1] != r_trt[1]
    ):
        return "TRT_VERSION_MISMATCH"

    if header["os"] != runtime["os"] or header["arch"] != runtime["arch"]:
        return "PLATFORM_MISMATCH"

    h_sm = header["sm"]
    r_sm = runtime["sm"]

    if h_sm > r_sm:
        return "INCOMPATIBLE_HARDWARE"
    if h_sm != r_sm and not header["hardware_compatible"]:
        return "INCOMPATIBLE_HARDWARE"

    return "SUCCESS"


def compute_penalty(header: dict, runtime: dict) -> float:
    status = classify_deserialization(header, runtime)
    if status != "SUCCESS":
        return float("inf")

    h_sm = header["sm"]
    r_sm = runtime["sm"]

    if h_sm == r_sm:
        penalty = 1.0
    else:
        penalty = 1.0 + 0.15 * (r_sm[0] - h_sm[0]) + 0.05 * (r_sm[1] - h_sm[1])

    if runtime.get("lean_runtime", False) != header.get("lean_runtime", False):
        penalty += 0.10

    return round(penalty, 4)
