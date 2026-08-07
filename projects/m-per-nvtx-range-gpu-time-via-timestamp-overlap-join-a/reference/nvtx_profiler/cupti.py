import sqlite3


def reproduce_cuda_api_sum(db_path):
    """Reproduce cuda_api_sum aggregation from raw CUPTI rows."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name, start_ns, end_ns FROM cupti_api_events")
    rows = cur.fetchall()
    conn.close()

    groups = {}
    for name, start_ns, end_ns in rows:
        dur = end_ns - start_ns
        if name not in groups:
            groups[name] = []
        groups[name].append(dur)

    result = {}
    for name, durs in groups.items():
        cnt = len(durs)
        tot = sum(durs)
        avg = float(tot) / cnt if cnt > 0 else 0.0
        mn = min(durs) if cnt > 0 else 0
        mx = max(durs) if cnt > 0 else 0
        result[name] = {
            "count": cnt,
            "total": tot,
            "avg": avg,
            "min": mn,
            "max": mx,
        }
    return result
