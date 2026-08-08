def analyze_kernel_entry(entry: dict) -> dict:
    """Extract grid, block and TFLOPS from torch.profiler trace entry."""
    args = entry.get("args", {})
    grid = tuple(args.get("grid", [1, 1, 1]))
    block = tuple(args.get("block", [1, 1, 1]))
    dur_us = float(entry.get("dur", 1.0))
    flops = float(args.get("flops", 0.0))

    tflops = flops / (dur_us * 1e6) if dur_us > 0 else 0.0

    return {
        "grid": grid,
        "block": block,
        "tflops": tflops
    }
