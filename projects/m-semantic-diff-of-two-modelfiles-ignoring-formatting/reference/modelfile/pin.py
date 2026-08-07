def build_deterministic_modelfile(base_model: str) -> str:
    return f"FROM {base_model}\nPARAMETER temperature 0.0\nPARAMETER top_p 1.0\nPARAMETER seed 42\n"

def verify_deterministic(text: str) -> bool:
    params = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] == "#":
            continue
        parts = line.split(maxsplit=1)
        if parts[0].upper() == "PARAMETER" and len(parts) > 1:
            p_parts = parts[1].split(maxsplit=1)
            if len(p_parts) == 2:
                params[p_parts[0].lower()] = p_parts[1].strip()
    try:
        temp = float(params.get("temperature", "-1"))
        has_seed = "seed" in params
        top_p = float(params.get("top_p", "-1"))
        return temp == 0.0 and has_seed and top_p == 1.0
    except ValueError:
        return False
