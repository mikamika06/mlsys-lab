def _parse_lines(text):
    instructions = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] == "#":
            continue
        parts = line.split(maxsplit=1)
        verb = parts[0].upper()
        args = parts[1] if len(parts) > 1 else ""
        if verb == "PARAMETER":
            arg_parts = args.split()
            if len(arg_parts) >= 2:
                args = f"{arg_parts[0].lower()} {arg_parts[1]}"
        instructions.append((verb, args))
    return instructions

def semantic_diff(mf1: str, mf2: str) -> dict:
    set1 = set(_parse_lines(mf1))
    set2 = set(_parse_lines(mf2))
    added = sorted(list(set2 - set1))
    removed = sorted(list(set1 - set2))
    return {
        "added": [{"verb": v, "args": a} for v, a in added],
        "removed": [{"verb": v, "args": a} for v, a in removed]
    }
