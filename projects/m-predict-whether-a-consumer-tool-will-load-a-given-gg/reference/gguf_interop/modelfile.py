def generate_modelfile(gguf_path: str, params: dict) -> str:
    lines = [f"FROM {gguf_path}"]

    template = params.get("template")
    if template:
        lines.append(f'TEMPLATE """{template}"""')

    system = params.get("system")
    if system:
        lines.append(f'SYSTEM """{system}"""')

    parameters = params.get("parameters", {})
    for k, v in sorted(parameters.items()):
        if isinstance(v, bool):
            v_str = "true" if v else "false"
        elif isinstance(v, list):
            for item in v:
                lines.append(f'PARAMETER {k} "{item}"')
            continue
        else:
            v_str = str(v)
        lines.append(f"PARAMETER {k} {v_str}")

    adapter = params.get("adapter")
    if adapter:
        lines.append(f"ADAPTER {adapter}")

    return "\n".join(lines) + "\n"
