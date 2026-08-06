def _parse_version(v_str: str) -> tuple:
    return tuple(int(x) for x in v_str.split("."))


def diagnose_portability_failure(host_profile: dict, binary_spec: dict) -> dict:
    """Diagnoses host compatibility issues for a given llamafile binary spec."""
    host_arch = host_profile.get("arch")
    supported_arches = binary_spec.get("supported_arches", [])
    if host_arch not in supported_arches:
        return {
            "is_compatible": False,
            "status": "UNSUPPORTED_ARCH",
            "missing_flags": [],
        }

    req_flags = set(binary_spec.get("required_cpu_flags", []))
    host_flags = set(host_profile.get("cpu_flags", []))
    missing_flags = sorted(list(req_flags - host_flags))
    if missing_flags:
        return {
            "is_compatible": False,
            "status": "MISSING_CPU_ISA",
            "missing_flags": missing_flags,
        }

    host_page = host_profile.get("page_size_kb")
    supp_pages = binary_spec.get("supported_page_sizes_kb", [])
    if supp_pages and host_page not in supp_pages:
        return {
            "is_compatible": False,
            "status": "INCOMPATIBLE_PAGE_SIZE",
            "missing_flags": [],
        }

    min_k = binary_spec.get("min_kernel_version")
    host_k = host_profile.get("kernel_version")
    if min_k and host_k:
        if _parse_version(host_k) < _parse_version(min_k):
            return {
                "is_compatible": False,
                "status": "OUTDATED_KERNEL",
                "missing_flags": [],
            }

    return {
        "is_compatible": True,
        "status": "COMPATIBLE",
        "missing_flags": [],
    }
