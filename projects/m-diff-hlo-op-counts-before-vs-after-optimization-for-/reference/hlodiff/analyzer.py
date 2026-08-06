import re


def parse_hlo_ops(hlo_text):
    counts = {}
    for line in hlo_text.splitlines():
        line = line.strip()
        if "=" in line and ("(" in line or "%" in line):
            parts = line.split("=")
            if len(parts) > 1:
                rhs = parts[1].strip()
                op_match = re.match(r"^([a-zA-Z0-9_-]+)", rhs)
                if op_match:
                    op = op_match.group(1)
                    if op not in ("fusion", "ROOT"):
                        counts[op] = counts.get(op, 0) + 1
    return counts


def diff_op_counts(before_text, after_text):
    b_counts = parse_hlo_ops(before_text)
    a_counts = parse_hlo_ops(after_text)
    all_ops = set(b_counts.keys()).union(set(a_counts.keys()))
    diffs = {}
    for op in all_ops:
        before_val = b_counts.get(op, 0)
        after_val = a_counts.get(op, 0)
        diffs[op] = {"before": before_val, "after": after_val, "delta": after_val - before_val}
    return diffs


def count_fusions(optimized_text):
    fusion_count = 0
    fused_ops = {}
    current_fusion = None
    for line in optimized_text.splitlines():
        line = line.strip()
        if "fusion(" in line or "is_fusion=true" in line or "kind=kFused" in line or "fusion" in line.lower():
            if "fuzzy" not in line.lower():
                fusion_count += 1
        if "fused_computation" in line or "fusion" in line:
            m = re.search(r"%([a-zA-Z0-9_.-]+)\s*=", line)
            if m:
                current_fusion = m.group(1)
                fused_ops[current_fusion] = 0
        elif current_fusion and "=" in line:
            fused_ops[current_fusion] += 1
    return {"fusion_kernels": fusion_count, "fused_blocks": len(fused_ops)}


def analyze_growth(dumps_by_size):
    results = []
    for size_name, (b_text, a_text) in sorted(dumps_by_size.items(), key=lambda x: len(x[0])):
        b_bytes = len(b_text.encode("utf-8"))
        a_bytes = len(a_text.encode("utf-8"))
        results.append({
            "size": size_name,
            "before_bytes": b_bytes,
            "after_bytes": a_bytes,
            "size_ratio": float(a_bytes) / float(b_bytes) if b_bytes > 0 else 0.0
        })
    return results
