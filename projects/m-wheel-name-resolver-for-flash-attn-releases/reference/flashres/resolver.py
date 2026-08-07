import re

def parse_wheel_name(filename: str) -> dict:
    name = filename[:-4] if filename.endswith(".whl") else filename
    parts = name.split("-")
    if len(parts) < 5:
        raise ValueError(f"Invalid wheel filename structure: {filename}")
    distribution = parts[0]
    version = parts[1]
    build = parts[2] if parts[2][0].isdigit() and parts[2] not in parts[1] else None
    offset = 2 if build else 1
    py_tag = parts[offset + 1]
    abi_tag = parts[offset + 2]
    plat_tag = "-".join(parts[offset + 3:])
    return {
        "distribution": distribution,
        "version": version,
        "build": build,
        "py_tag": py_tag,
        "abi_tag": abi_tag,
        "plat_tag": plat_tag,
    }
