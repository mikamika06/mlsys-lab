SAMPLES = [
    (
        "FROM llama3\n"
        "# This is a comment\n"
        "PARAMETER temperature 0.7\n"
        'PARAMETER stop "</s>"\n'
        'PARAMETER stop "<|eot_id|>"\n'
        "SYSTEM \"\"\"\nYou are a helpful assistant.\nKeep it concise.\n\"\"\"\n",
        {
            "from": "llama3",
            "parameters": {
                "temperature": "0.7",
                "stop": ["</s>", "<|eot_id|>"]
            },
            "system": "You are a helpful assistant.\nKeep it concise."
        }
    ),
    (
        "FROM mistral:7b\n"
        "PARAMETER num_ctx 4096\n"
        'PARAMETER stop "User:"\n'
        'PARAMETER stop "Assistant:"\n',
        {
            "from": "mistral:7b",
            "parameters": {
                "num_ctx": "4096",
                "stop": ["User:", "Assistant:"]
            },
            "system": None
        }
    )
]

def parse_modelfile(content):
    import re
    lines = content.splitlines()
    res = {"from": None, "parameters": {}, "system": None}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if line.startswith("FROM "):
            res["from"] = line[5:].strip()
            i += 1
        elif line.startswith("PARAMETER "):
            parts = line[10:].strip().split(None, 1)
            if len(parts) == 2:
                pkey, pval = parts[0], parts[1].strip('"\'')
                if pkey == "stop":
                    res["parameters"].setdefault("stop", []).append(pval)
                else:
                    res["parameters"][pkey] = pval
            i += 1
        elif line.startswith("SYSTEM "):
            rest = line[7:].strip()
            if rest.startswith('"""'):
                if rest.endswith('"""') and len(rest) >= 6:
                    res["system"] = rest[3:-3].strip("\n")
                    i += 1
                else:
                    block_lines = [rest[3:]]
                    i += 1
                    while i < len(lines):
                        if '"""' in lines[i]:
                            block_lines.append(lines[i].split('"""')[0])
                            i += 1
                            break
                        else:
                            block_lines.append(lines[i])
                            i += 1
                    res["system"] = "\n".join(block_lines).strip("\n")
            else:
                res["system"] = rest.strip('"\'')
                i += 1
        else:
            i += 1
    return res

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
