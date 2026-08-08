def analyze_torch(events, flops_per_thread):
    tflops_sums = {}
    counts = {}
    for ev in events:
        if ev.get("cat") == "kernel":
            name = ev["name"]
            args = ev.get("args", {})
            dur_us = ev.get("dur", 0.0)
            if name in flops_per_thread and dur_us > 0:
                grid = args.get("Grid X", 1) * args.get("Grid Y", 1) * args.get("Grid Z", 1)
                block = args.get("Block X", 1) * args.get("Block Y", 1) * args.get("Block Z", 1)
                threads = grid * block
                flops = threads * flops_per_thread[name]
                tflops = flops / (dur_us * 1e-6) / 1e12
                tflops_sums[name] = tflops_sums.get(name, 0.0) + tflops
                counts[name] = counts.get(name, 0) + 1

    return {k: tflops_sums[k] / counts[k] for k in counts}
