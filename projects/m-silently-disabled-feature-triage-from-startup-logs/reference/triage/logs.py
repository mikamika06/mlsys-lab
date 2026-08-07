import re

def parse_startup_logs(log_text):
    disabled = []
    patterns = [
        r"WARNING.*falling back to.*for (\w+)",
        r"INFO.*disabled (\w+) due to (\w+)",
        r"NOTICE.*(\w+) unavailable, using fallback",
    ]
    for line in log_text.splitlines():
        for pat in patterns:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                feature = m.group(1).lower()
                if feature not in disabled:
                    disabled.append(feature)
    return sorted(disabled)
