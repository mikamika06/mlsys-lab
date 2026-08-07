VALID_VERBS = {"FROM", "PARAMETER", "TEMPLATE", "SYSTEM", "LICENSE", "MESSAGE"}
VALID_PARAMS = {"temperature", "top_p", "top_k", "seed", "num_ctx", "repeat_penalty"}

def validate_modelfile(text: str):
    for idx, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line[0] == "#":
            continue
        parts = line.split(maxsplit=1)
        verb = parts[0].upper()
        if verb not in VALID_VERBS:
            return False, idx, f"Invalid instruction verb: {verb}"
        if verb == "PARAMETER":
            if len(parts) < 2:
                return False, idx, "PARAMETER missing name and value"
            p_parts = parts[1].split()
            if not p_parts or p_parts[0].lower() not in VALID_PARAMS:
                p_name = p_parts[0] if p_parts else ""
                return False, idx, f"Invalid parameter name: {p_name}"
    return True, None, None
