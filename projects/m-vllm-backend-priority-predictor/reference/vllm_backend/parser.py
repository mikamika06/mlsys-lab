import re


def parse_startup_logs(log_text):
    selected = None
    rejected = {}
    candidates = []

    m_sel = re.search(r"Using ([A-Z0-9_]+) backend", log_text)
    if m_sel:
        selected = m_sel.group(1)

    m_cand = re.search(r"Candidate backends: \[([^\]]+)\]", log_text)
    if m_cand:
        candidates = [c.strip().strip("'\"") for c in m_cand.group(1).split(",") if c.strip()]

    for line in log_text.splitlines():
        m_rej = re.search(r"Cannot use ([A-Z0-9_]+) backend: (.*)", line)
        if m_rej:
            backend = m_rej.group(1)
            reason = m_rej.group(2).strip()
            rejected[backend] = reason

    return {
        "selected": selected,
        "candidates": candidates,
        "rejected": rejected
    }


def extract_rejection_reasons(log_lines):
    rejections = []
    for line in log_lines:
        m = re.search(r"Cannot use ([A-Z0-9_]+) backend: (.*)", line)
        if m:
            rejections.append((m.group(1), m.group(2).strip()))
    return rejections
