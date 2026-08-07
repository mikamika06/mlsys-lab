def emit_modelfile(data: dict) -> str:
    lines = []
    for inst, val in data.get("instructions", []):
        if "\n" in val or '"""' in val:
            lines.append(f'{inst} """')
            lines.append(val)
            lines.append('"""')
        else:
            lines.append(f"{inst} {val}")

    for k, v in data.get("parameters", {}).items():
        lines.append(f"PARAMETER {k} {v}")

    for s in data.get("stops", []):
        lines.append(f"PARAMETER stop {s}")

    return "\n".join(lines) + "\n"
