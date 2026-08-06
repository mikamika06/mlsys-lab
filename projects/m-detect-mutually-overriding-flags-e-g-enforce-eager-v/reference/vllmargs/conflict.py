def detect_conflicts(args):
    out = []
    if args.get("enforce_eager") and args.get("cudagraph_capture_sizes"):
        out.append("enforce_eager conflicts with cudagraph_capture_sizes")
    if args.get("disable_sliding_window") and args.get("sliding_window"):
        out.append("disable_sliding_window conflicts with sliding_window")
    if args.get("kv_cache_dtype") == "fp8" and args.get("enable_prefix_caching") and args.get("no_prefix_caching"):
        out.append("enable_prefix_caching conflicts with no_prefix_caching")
    return out
