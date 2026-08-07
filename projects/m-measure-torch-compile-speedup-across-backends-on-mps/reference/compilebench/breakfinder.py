def identify_graph_break(log_lines):
    for line in log_lines:
        if "graph_break" in line or "Graph break" in line or "reason:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                return parts[-1].strip()
            return line.strip()
    return "none"
