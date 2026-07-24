def assign_layers(layer_sizes, gpu_caps, cpu_cap):
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
