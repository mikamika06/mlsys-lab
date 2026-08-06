def diff_op_counts(before_text, after_text):
    """Compute difference in operation counts between before and after HLO texts."""
    def parse_ops(text):
        counts = {}
        for line in text.splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("HloModule") and not line.startswith("ENTRY") and "{" not in line and "}" not in line:
                parts = line.split("=")
                if len(parts) == 2:
                    rhs = parts[1].strip()
                    op_name = rhs.split("(")[0].strip().split()[0]
                    counts[op_name] = counts.get(op_name, 0) + 1
        return counts
    b_counts = parse_ops(before_text)
    a_counts = parse_ops(after_text)
    all_keys = set(b_counts.keys()).union(set(a_counts.keys()))
    delta = {}
    for k in all_keys:
        delta[k] = a_counts.get(k, 0) - b_counts.get(k, 0)
    return {"before": b_counts, "after": a_counts, "delta": delta}
