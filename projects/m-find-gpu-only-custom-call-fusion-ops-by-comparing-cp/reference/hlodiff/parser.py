def parse_hlo_ops(text):
    ops = []
    for line in text.splitlines():
        line = line.strip()
        if "custom-call" in line or "fusion" in line:
            ops.append(line)
    return ops
