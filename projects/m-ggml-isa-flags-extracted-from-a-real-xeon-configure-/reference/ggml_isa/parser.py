def parse_isa_flags(log_str):
    res = {}
    for token in log_str.split():
        if token.startswith("-DGGML_"):
            parts = token[2:].split("=")
            if len(parts) == 2:
                key, val = parts
                res[key] = (val.upper() == "ON")
    return res
