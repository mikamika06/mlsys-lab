from gpuhlo.analyzer import parse_hlo_module, extract_computations_and_ops


def find_gpu_only_ops(cpu_hlo, gpu_hlo):
    cpu_mod = parse_hlo_module(cpu_hlo)
    gpu_mod = parse_hlo_module(gpu_hlo)
    cpu_ops = extract_computations_and_ops(cpu_mod)
    gpu_ops = extract_computations_and_ops(gpu_mod)
    cpu_signatures = {(o["computation"], o["op"], o["expression"]) for o in cpu_ops}
    gpu_signatures = {(o["computation"], o["op"], o["expression"]) for o in gpu_ops}
    diff = gpu_signatures - cpu_signatures
    gpu_only = []
    for item in sorted(list(diff)):
        comp, op_name, expr = item
        if op_name in ("custom-call", "fusion", "bitcast", "transpose"):
            gpu_only.append({"computation": comp, "op": op_name, "expression": expr})
    return gpu_only


def verify_shape_consistency(cpu_hlo, gpu_hlo):
    cpu_mod = parse_hlo_module(cpu_hlo)
    gpu_mod = parse_hlo_module(gpu_hlo)
    cpu_ops = extract_computations_and_ops(cpu_mod)
    gpu_ops = extract_computations_and_ops(gpu_mod)
    cpu_shapes = {o["expression"] for o in cpu_ops}
    gpu_shapes = {o["expression"] for o in gpu_ops}
    return len(cpu_shapes.intersection(gpu_shapes)) > 0
