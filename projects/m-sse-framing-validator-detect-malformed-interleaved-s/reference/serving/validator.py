def validate_sse_stream(lines):
    if not isinstance(lines, list):
        return False
    for line in lines:
        if line == "":
            continue
        if line.startswith(":"):
            continue
        if not any(line.startswith(prefix) for prefix in ("data:", "event:", "id:", "retry:")):
            return False
    return True
