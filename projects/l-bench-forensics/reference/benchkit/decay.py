from .parse import derive, kind
from .stats import median, separable


def by_depth(rows, model=None, want="decode"):
    out = {}
    for r in rows:
        if kind(r) != want:
            continue
        if model and r.get("model_type") != model:
            continue
        out[int(r.get("n_depth", 0))] = derive(r)
    return dict(sorted(out.items()))


def decay_table(rows, model=None, want="decode"):
    table = by_depth(rows, model, want)
    if 0 not in table:
        return []
    base = table[0]
    base_ts = median(base["samples_ts"]) if base["samples_ts"] else base["reported_ts"]
    out = []
    for depth, row in table.items():
        ts = median(row["samples_ts"]) if row["samples_ts"] else row["reported_ts"]
        out.append({
            "depth": depth,
            "tokens_per_second": ts,
            "relative_to_empty": ts / base_ts if base_ts else 0.0,
            "loss_fraction": 1.0 - (ts / base_ts if base_ts else 0.0),
            "separable_from_empty": separable(row["samples_ts"], base["samples_ts"]),
        })
    return out


def slope_per_1k(table):
    """Least-squares tokens/s lost per 1024 tokens of context."""
    pts = [(t["depth"], t["tokens_per_second"]) for t in table]
    if len(pts) < 2:
        return 0.0
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    if not den:
        return 0.0
    return (n * sxy - sx * sy) / den * 1024.0


def extrapolate(table, depth):
    if not table:
        return 0.0
    slope = slope_per_1k(table) / 1024.0
    base = table[0]
    return base["tokens_per_second"] + slope * (depth - base["depth"])
