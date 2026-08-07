import re

def extract_shape_guards(guard_logs):
    guards = []
    pattern = re.compile(r"L\['(\w+)'\]\.size\(\)\[(\d+)\]\s*==\s*(\d+)")
    for line in guard_logs.splitlines():
        if "GUARD" in line or "size()" in line:
            matches = pattern.findall(line)
            for var, dim, val in matches:
                expr = f"L['{var}'].size()[{dim}] == {val}"
                guards.append({
                    "var": var,
                    "dim": int(dim),
                    "val": int(val),
                    "expr": expr
                })
    return guards
