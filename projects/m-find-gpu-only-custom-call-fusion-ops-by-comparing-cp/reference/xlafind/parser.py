import re

def parse_hlo_dump(hlo_text):
    ops = {}
    current_op = None
    for line in hlo_text.splitlines():
        line_s = line.strip()
        match = re.match(r'^%([a-zA-Z0-9_.-]+)\s*=\s*([a-zA-Z0-9_-]+)', line_s)
        if match:
            op_name, op_type = match.groups()
            current_op = op_name
            ops[op_name] = {"type": op_type, "text": line_s}
        elif current_op and line_s:
            ops[current_op]["text"] += "\n" + line_s
    return ops

def find_gpu_only_ops(cpu_hlo, gpu_hlo):
    cpu_ops = parse_hlo_dump(cpu_hlo)
    gpu_ops = parse_hlo_dump(gpu_hlo)
    gpu_only = []
    for name, data in gpu_ops.items():
        if name not in cpu_ops or data["type"] in ("custom-call", "fusion"):
            if name not in cpu_ops:
                gpu_only.append((name, data["type"]))
            elif data["type"] in ("custom-call", "fusion") and cpu_ops[name]["text"] != data["text"]:
                gpu_only.append((name, data["type"]))
    return sorted(gpu_only)
