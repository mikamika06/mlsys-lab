import re

FIXTURES = [
    """FROM llama3
# A simple comment
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature "0.7"
SYSTEM \"\"\"
You are a helpful assistant.
\"\"\"
TEMPLATE "<|start_header_id|>{{ .Role }}<|end_header_id|>{{ .Content }}<|eot_id|>"
""",
    """FROM mistral
PARAMETER stop "<|im_end|>"
MESSAGE user "hello"
MESSAGE assistant "hi"
TEMPLATE "<|im_start|>user\\n{{ .Prompt }}<|im_end|>\\n<|im_start|>assistant\\n"
"""
]

def parse(text: str) -> dict:
    ast = {"PARAMETER": {}, "MESSAGE": []}
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0].upper()
        val = parts[1] if len(parts) > 1 else ""
        if val.startswith('"""'):
            if val.endswith('"""') and len(val) >= 6:
                val = val[3:-3]
            else:
                val = val[3:] + "\n"
                i += 1
                while i < len(lines) and '"""' not in lines[i]:
                    val += lines[i] + "\n"
                if i < len(lines):
                    val += lines[i].replace('"""', '')
        elif val.startswith('"') and val.endswith('"') and len(val) >= 2:
            val = val[1:-1]

        if cmd == "PARAMETER":
            p_parts = val.split(maxsplit=1)
            pk = p_parts[0]
            pv = p_parts[1] if len(p_parts) > 1 else ""
            if pv.startswith('"') and pv.endswith('"') and len(pv) >= 2:
                pv = pv[1:-1]
            ast["PARAMETER"].setdefault(pk, []).append(pv)
        elif cmd == "MESSAGE":
            m_parts = val.split(maxsplit=1)
            r = m_parts[0]
            c = m_parts[1] if len(m_parts) > 1 else ""
            if c.startswith('"') and c.endswith('"') and len(c) >= 2:
                c = c[1:-1]
            ast["MESSAGE"].append({"role": r, "content": c})
        else:
            ast[cmd] = val
        i += 1
    return ast

def dumps(ast: dict) -> str:
    lines = []
    if "FROM" in ast:
        lines.append(f'FROM "{ast["FROM"]}"')
    for k, v in ast.items():
        if k in ("PARAMETER", "MESSAGE", "FROM"):
            continue
        if "\n" in v:
            lines.append(f'{k} """{v}"""')
        else:
            lines.append(f'{k} "{v}"')

    for k, vals in ast.get("PARAMETER", {}).items():
        for v in vals:
            lines.append(f'PARAMETER {k} "{v}"')

    for m in ast.get("MESSAGE", []):
        lines.append(f'MESSAGE {m["role"]} "{m["content"]}"')

    return "\n".join(lines)

def find_missing_stops(ast: dict) -> list:
    template = ast.get("TEMPLATE", "")
    tokens = set(re.findall(r"<\|.*?\|>", template))
    stops = set(ast.get("PARAMETER", {}).get("stop", []))
    return sorted(list(tokens - stops))
