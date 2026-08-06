from hlodiff.parser import parse_hlo_ops


def find_gpu_only_ops(cpu_text, gpu_text):
    cpu_ops = set(parse_hlo_ops(cpu_text))
    gpu_ops = set(parse_hlo_ops(gpu_text))
    return sorted(list(gpu_ops - cpu_ops))
