import re

def analyze_objdump(lines: list[str]) -> dict:
    counts = {"vpmaddubsw": 0, "vpmaddwd": 0, "vpaddd": 0, "vpdpbusd": 0}
    for line in lines:
        words = re.findall(r'[a-zA-Z0-9_]+', line.lower())
        for w in words:
            if w in counts:
                counts[w] += 1
    return counts
