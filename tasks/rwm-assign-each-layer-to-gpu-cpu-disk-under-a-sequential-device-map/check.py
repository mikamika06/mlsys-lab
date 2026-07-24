import random

def _oracle(layer_sizes, gpu_caps, cpu_cap):
    rem_gpu = list(gpu_caps)
    rem_cpu = cpu_cap
    assignments = []
    for s in layer_sizes:
        assigned = False
        for i, cap in enumerate(rem_gpu):
            if cap >= s:
                assignments.append(f"gpu{i}")
                rem_gpu[i] -= s
                assigned = True
                break
        if not assigned:
            if rem_cpu >= s:
                assignments.append("cpu")
                rem_cpu -= s
                assigned = True
            else:
                assignments.append("disk")
    return assignments

def grade(sol, fx) -> dict:
    ok = 1.0
    for _ in range(5):
        n_layers = random.randint(1, 20)
        layer_sizes = [random.randint(10, 1000) for _ in range(n_layers)]
        n_gpus = random.randint(1, 4)
        gpu_caps = [random.randint(500, 2000) for _ in range(n_gpus)]
        cpu_cap = random.randint(2000, 8000)
        try:
            got = sol.assign_layers(layer_sizes, gpu_caps, cpu_cap)
        except Exception:
            return {"exact_match": 0.0}
        ref = _oracle(layer_sizes, gpu_caps, cpu_cap)
        if list(got) != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
