def parse_kernel(code: str):
    lines = [l.strip() for l in code.splitlines() if l.strip()]
    ops = []
    for line in lines:
        if "=" in line:
            lhs, rhs = line.split("=", 1)
            ops.append({"lhs": lhs.strip(), "rhs": rhs.strip()})
    return ops
