import re


def parse_graph_break_log(log_text):
    records = []
    lines = log_text.strip().splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("GRAPH BREAK:") or line.startswith("[GRAPH BREAK]"):
            loc_match = re.search(r'file\s+"([^"]+)",\s+line\s+(\d+)', line)
            if not loc_match:
                loc_match = re.search(r'([a-zA-Z0-9_/\.\-]+):(\d+)', line)
            filename = loc_match.group(1) if loc_match else "unknown"
            lineno = int(loc_match.group(2)) if loc_match else 0

            reason_match = re.search(r"reason:\s*(.*)", line, re.IGNORECASE)
            reason = reason_match.group(1).strip() if reason_match else line.strip()

            stack = []
            i += 1
            while i < n and (lines[i].startswith("  ") or lines[i].startswith("\t") or lines[i].startswith("    at ")):
                stack.append(lines[i].strip())
                i += 1

            records.append({
                "filename": filename,
                "lineno": lineno,
                "reason": reason,
                "stack": stack
            })
        else:
            i += 1
    return records


def summarize_breaks(parsed_records):
    summary = {}
    for rec in parsed_records:
        reason = rec["reason"]
        summary[reason] = summary.get(reason, 0) + 1
    return summary
