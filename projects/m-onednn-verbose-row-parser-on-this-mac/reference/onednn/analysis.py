from onednn.parser import parse_row

def analyze_log(rows, wall_time):
    parsed = [parse_row(r) for r in rows if parse_row(r) is not None]
    classes = {}
    total_kernel_time = sum(p["time_ms"] for p in parsed)
    for p in parsed:
        impl = p["impl"]
        base = impl.split(":")[0]
        classes.setdefault(base, 0.0)
        classes[base] += p["time_ms"]
    ratio = total_kernel_time / wall_time if wall_time > 0 else 0.0
    return {
        "total_kernel_time_ms": round(total_kernel_time, 2),
        "wall_time_ms": round(wall_time, 2),
        "ratio": round(ratio, 4),
        "classes": {k: round(v, 2) for k, v in classes.items()}
    }
