def parse_logs(lines):
    res = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        d = {}
        for p in parts[1:]:
            k, v = p.split("=")
            d[k] = float(v) if "." in v or "E" in v else int(v)
        res.append(d)
    return res
