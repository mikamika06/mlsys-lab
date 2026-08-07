def parse_verbose_line(line):
    line = line.strip()
    if not line.startswith("onednn_verbose,"):
        return None
    parts = line.split(",")
    if len(parts) < 10:
        return None
    event_type = parts[1].strip()
    if event_type != "exec":
        return None
    engine = parts[2].strip()
    prim_kind = parts[3].strip()
    impl = parts[4].strip()
    prop_kind = parts[5].strip()
    shapes = parts[6].strip()
    in_layout = parts[7].strip()
    out_layout = parts[8].strip()
    try:
        exec_time_ms = float(parts[9].strip())
    except ValueError:
        return None
    return {
        "engine": engine,
        "prim_kind": prim_kind,
        "impl": impl,
        "prop_kind": prop_kind,
        "shapes": shapes,
        "in_layout": in_layout,
        "out_layout": out_layout,
        "exec_time_ms": exec_time_ms,
    }


def parse_verbose_log(log_text):
    records = []
    for line in log_text.strip().splitlines():
        rec = parse_verbose_line(line)
        if rec is not None:
            records.append(rec)
    return records
