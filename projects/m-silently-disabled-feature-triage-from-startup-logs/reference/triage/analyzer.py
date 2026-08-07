import re

def parse_startup_logs(log_lines):
    pattern = re.compile(r"silently disabled feature:\s*([a-zA-Z0-9_-]+)|warning:\s*([a-zA-Z0-9_-]+)\s*disabled", re.IGNORECASE)
    disabled = []
    for line in log_lines:
        match = pattern.search(line)
        if match:
            feat = next(g for g in match.groups() if g is not None)
            if feat not in disabled:
                disabled.append(feat)
    return sorted(disabled)
