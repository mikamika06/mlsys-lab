VALID_INSTRUCTIONS = {
    "FROM",
    "PARAMETER",
    "SYSTEM",
    "TEMPLATE",
    "ADAPTER",
    "LICENSE",
    "MESSAGE",
}


def validate_modelfile(content: str) -> tuple[bool, int, str]:
    """Validates Modelfile lines against instruction rules and returns first failure."""
    lines = content.splitlines()
    has_from = False
    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        instr = parts[0].upper()
        if instr not in VALID_INSTRUCTIONS:
            return (False, idx, f"Unknown instruction '{parts[0]}'")
        if len(parts) < 2:
            return (False, idx, f"Missing arguments for instruction '{instr}'")
        arg = parts[1].strip()
        if instr == "FROM":
            has_from = True
            if not arg:
                return (False, idx, "FROM requires a valid base model reference")
        elif instr == "PARAMETER":
            p_parts = arg.split(maxsplit=1)
            if len(p_parts) < 2:
                return (False, idx, "PARAMETER requires a key and a value")
        elif instr == "MESSAGE":
            m_parts = arg.split(maxsplit=1)
            if len(m_parts) < 2:
                return (False, idx, "MESSAGE requires a role and content")
    if not has_from:
        return (False, 0, "Missing required FROM instruction")
    return (True, 0, "OK")
