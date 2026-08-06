import re

def parse_top_5(log_text: str) -> list:
    layers = []
    for line in log_text.splitlines():
        if " EXECUTED " in line or " NOT_RUN " in line:
            m = re.search(r"\[ INFO \]\s+(\S+)\s+(?:EXECUTED|NOT_RUN|OPTIMIZED_OUT)\s+\S+\s+\S+\s+([\d\.]+)\s+([\d\.]+)", line)
            if m:
                layers.append((m.group(1), float(m.group(2))))
    layers.sort(key=lambda x: x[1], reverse=True)
    return layers[:5]
