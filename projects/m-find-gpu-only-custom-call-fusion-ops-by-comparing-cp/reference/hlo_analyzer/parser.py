import re

def parse_hlo(hlo_text):
    ops = []
    for line in hlo_text.splitlines():
        line = line.strip()
        if not line or line.startswith("HloModule") or (line.startswith("%") and "->" in line):
            continue
        m_custom = re.search(r'(custom-call)\s*([^\(\s]+)', line)
        m_fusion = re.search(r'(fusion)\s*([^\(\s]+)', line)
        if m_custom:
            ops.append({"type": "custom-call", "name": m_custom.group(2)})
        elif m_fusion:
            ops.append({"type": "fusion", "name": m_fusion.group(2)})
        elif "custom-call" in line:
            ops.append({"type": "custom-call", "name": "unknown"})
        elif "fusion" in line:
            ops.append({"type": "fusion", "name": "unknown"})
    return ops
