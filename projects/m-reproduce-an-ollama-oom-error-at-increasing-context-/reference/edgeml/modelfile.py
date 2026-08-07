def parse_modelfile(text):
    system = None
    quant = None
    from_base = None
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("FROM"):
            from_base = line.split(maxsplit=1)[1].strip()
            if "-" in from_base:
                parts = from_base.split("-")
                quant = parts[-1].lower()
        elif line.startswith("SYSTEM"):
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                system = parts[1].strip().strip('"').strip("'")
    return {"from": from_base, "system": system, "quant": quant}


def verify_modelfile(parsed, expected_system, expected_quant):
    s_match = parsed.get("system") == expected_system
    q_match = parsed.get("quant") == expected_quant.lower()
    return bool(s_match and q_match)
