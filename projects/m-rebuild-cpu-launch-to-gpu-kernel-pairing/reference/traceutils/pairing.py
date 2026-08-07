def pair_launches(events):
    """Pair CPU launch events with GPU kernels using correlation ids."""
    cpu_launches = {}
    gpu_kernels = {}
    for ev in events:
        corr = ev.get("args", {}).get("correlation_id")
        if corr is not None:
            if ev.get("cat") == "cpu_op":
                cpu_launches[corr] = ev
            elif ev.get("cat") == "gpu_kernel":
                gpu_kernels[corr] = ev
    pairs = []
    for corr in sorted(cpu_launches.keys()):
        if corr in gpu_kernels:
            pairs.append((cpu_launches[corr]["name"], gpu_kernels[corr]["name"]))
    return pairs
