import re

def top_layers(text):
    rows = []
    for line in text.splitlines():
        if "|" in line and "ms" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                name = parts[1]
                time_str = parts[2].replace("ms", "").strip()
                try:
                    val = float(time_str)
                    rows.append((name, val))
                except ValueError:
                    pass
    rows.sort(key=lambda x: x[1], reverse=True)
    return [{"layer": r[0], "time_ms": r[1]} for r in rows[:5]]
