import re


def analyze_stablehlo_ir(ir_text: str) -> dict:
    op_pattern = re.compile(r"%[a-zA-Z0-9_%#:\.\-]+(?:\s*:\s*[^{=]+)?\s*=\s*(stablehlo\.[a-zA-Z0-9_]+)|^\s*(stablehlo\.[a-zA-Z0-9_]+)")
    counts = {}
    for line in ir_text.splitlines():
        line_str = line.strip()
        if not line_str or line_str.startswith("//"):
            continue
        match = op_pattern.search(line_str)
        if match:
            opname = match.group(1) or match.group(2)
            counts[opname] = counts.get(opname, 0) + 1
    return {
        "op_counts": counts,
        "unique_ops": sorted(counts.keys()),
    }
