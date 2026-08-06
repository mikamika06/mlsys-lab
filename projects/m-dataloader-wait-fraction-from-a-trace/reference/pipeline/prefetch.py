def simulate_double_buffer(load_times, h2d_times, compute_times):
    n = len(load_times)
    if n == 0:
        return 0.0
    t_gpu = 0.0
    t_cpu = 0.0
    for i in range(n):
        l = load_times[i]
        h = h2d_times[i]
        c = compute_times[i]
        if i == 0:
            t_cpu = l
            t_gpu = t_cpu + h + c
        else:
            t_cpu = max(t_cpu, t_cpu + l)
            t_gpu = max(t_gpu, t_cpu + h) + c
    return float(t_gpu)
