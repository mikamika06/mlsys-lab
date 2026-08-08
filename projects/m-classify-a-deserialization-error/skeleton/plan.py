def parse_header(engine_bytes: bytes) -> dict:
    """
    Parse the first 20 bytes of a .plan file.
    Format is 5 32-bit little-endian fields:
    - magic (4 bytes)
    - trt_version (uint32)
    - build_sm (uint32)
    - hw_compat (uint32)
    - os_id (uint32)

    Returns a dict with keys: 'magic', 'trt_version', 'build_sm', 'hw_compat', 'os_id'.
    """
    raise NotImplementedError


def diagnose_load(engine_bytes: bytes, env_trt: int, env_sm: int, env_os: int) -> dict:
    """
    Diagnoses engine load compatibility and performance penalty.
    Rules:
    - If engine_bytes is < 20 bytes, status="ERR_TRUNCATED"
    - If magic != b'TRT\x00', status="ERR_MAGIC"
    - If TRT version != env_trt, status="ERR_TRT_VERSION"
    - If OS ID != env_os, status="ERR_OS"
    - If build_sm != env_sm:
        - If hw_compat == 0, status="ERR_SM_ARCH"
        - If hw_compat == 1, but either build_sm < 80 or env_sm < 80, status="ERR_SM_ARCH_UNSUPPORTED"
        - Otherwise, status="OK", penalty=8.5
    - If build_sm == env_sm:
        - If hw_compat == 1, status="OK", penalty=3.0
        - If hw_compat == 0, status="OK", penalty=0.0

    All non-OK statuses have penalty=0.0.
    Returns: {"status": str, "penalty": float}
    """
    raise NotImplementedError
