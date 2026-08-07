import re

def parse_row(line: str) -> dict:
    match = re.search(r"oneDNN_verbose,\s*(?P<mode>\w+),\s*(?P<prim>[\w:]+),\s*(?P<impl>[^,]+),\s*(?P<prop>[\w:]*),\s*(?P<fmt>[^,]*),\s*(?P<name>[^,]*),\s*(?P<time>[\d\.]+)", line)
    if not match:
        return {}
    d = match.groupdict()
    return {
        "mode": d["mode"],
        "primitive": d["prim"],
        "impl": d["impl"].strip(),
        "prop": d["prop"],
        "format": d["fmt"].strip(),
        "name": d["name"].strip(),
        "time_ms": float(d["time"])
    }
