import re

EVENT = re.compile(r"^\[\d+/\d+\] Recompiling function", re.M)
GUARD = re.compile(r"^\s*-\s*\d+/\d+:\s*(.+?)\s*$", re.M)


def failing_guards(text: str):
    events = []
    positions = [m.start() for m in EVENT.finditer(text)]
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        events.append([g.strip() for g in GUARD.findall(text[start:end])])
    return events
