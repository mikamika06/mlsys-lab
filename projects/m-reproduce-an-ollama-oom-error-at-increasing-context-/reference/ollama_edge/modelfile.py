def parse_modelfile(content):
    lines = content.strip().split("\n")
    res = {"from": "", "system": "", "quant": ""}
    for line in lines:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            key, val = parts[0].lower(), parts[1].strip('"')
            if key == "from":
                res["from"] = val
            elif key == "system":
                res["system"] = val
            elif key == "quant":
                res["quant"] = val
    return res


def generate_modelfile(base_model, system_prompt, quant):
    return f"FROM {base_model}\nSYSTEM \"{system_prompt}\"\nQUANT {quant}\n"
