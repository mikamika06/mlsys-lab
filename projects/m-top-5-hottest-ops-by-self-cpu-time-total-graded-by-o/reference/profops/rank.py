from profops.parser import parse_profiler_table

def get_top_ops(rows, k=5):
    parsed = parse_profiler_table(rows)
    sorted_ops = sorted(parsed.items(), key=lambda x: x[1], reverse=True)
    return [op for op, _ in sorted_ops[:k]]
