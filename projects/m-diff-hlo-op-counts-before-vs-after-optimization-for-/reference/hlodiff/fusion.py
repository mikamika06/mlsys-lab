def count_fusion_kernels(hlo_text):
    """Count fusion kernels and extract kernel names from HLO text."""
    kernels = []
    for line in hlo_text.splitlines():
        line = line.strip()
        if "fusion(" in line or "fusion =" in line:
            if "calls=" in line:
                call_part = line.split("calls=")[1].strip().split(",")[0].strip()
                kernels.append(call_part)
            else:
                kernels.append("unnamed_fusion")
    return {"fusion_count": len(kernels), "kernel_names": sorted(list(set(kernels)))}
