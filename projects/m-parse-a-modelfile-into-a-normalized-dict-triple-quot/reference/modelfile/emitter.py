def emit_modelfile(data):
    lines = []
    if data.get("from"):
        lines.append(f"FROM {data['from']}")
    for k, v in data.get("parameters", {}).items():
        if isinstance(v, list):
            for item in v:
                lines.append(f"PARAMETER {k} \"{item}\"")
        else:
            lines.append(f"PARAMETER {k} {v}")
    sys_val = data.get("system")
    if sys_val:
        if "\n" in sys_val:
            lines.append('SYSTEM """')
            lines.append(sys_val)
            lines.append('"""')
        else:
            lines.append(f'SYSTEM "{sys_val}"')
    return "\n".join(lines) + "\n"
