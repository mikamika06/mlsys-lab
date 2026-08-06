from hlo_analyzer.parser import parse_hlo

def find_gpu_only_ops(cpu_text, gpu_text):
    cpu_ops = {(o["type"], o["name"]) for o in parse_hlo(cpu_text)}
    gpu_ops = {(o["type"], o["name"]) for o in parse_hlo(gpu_text)}
    diff = gpu_ops - cpu_ops
    return [{"type": t, "name": n} for t, n in sorted(diff)]
