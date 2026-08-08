def parse_nsys_summary(text):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return []
    headers = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        vals = [v.strip() for v in line.split(",")]
        row = {}
        for h, v in zip(headers, vals):
            if h in ("Total Time(ns)", "Active Time(ns)", "Instances", "Avg(ns)", "Med(ns)", "Min(ns)", "Max(ns)"):
                row[h] = float(v) if v else 0.0
            else:
                row[h] = v
        rows.append(row)
    return rows


def parse_cuda_api_sum(text):
    return parse_nsys_summary(text)
