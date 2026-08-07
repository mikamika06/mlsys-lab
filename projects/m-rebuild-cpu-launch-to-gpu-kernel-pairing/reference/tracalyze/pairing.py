def pair_launches(trace_data):
    events = trace_data.get("traceEvents", [])
    correlations = {}
    launches = {}
    kernels = {}
    for ev in events:
        args = ev.get("args", {})
        if "correlation" in args:
            corr = args["correlation"]
            name = ev.get("name", "")
            ph = ev.get("ph", "")
            if "enqueue" in name or "Launch" in name or ph == "X" and "cudaLaunch" in name:
                correlations[corr] = ev
            elif "kernel" in name.lower() or "cuLaunchKernel" in name:
                kernels[corr] = ev
    pairs = []
    for corr, launch_ev in correlations.items():
        if corr in kernels:
            pairs.append({"correlation": corr, "launch": launch_ev.get("name"), "kernel": kernels[corr].get("name")})
    return sorted(pairs, key=lambda x: x["correlation"])
