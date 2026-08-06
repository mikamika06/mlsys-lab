def parse_overhead_records(records):
    """Summarize timing records per workload into JIT and AOT overhead profiles."""
    grouped = {}
    for r in records:
        w = r["workload"]
        if w not in grouped:
            grouped[w] = {
                "jit_compile_ms": 0.0,
                "jit_execs": [],
                "aot_load_ms": 0.0,
                "aot_execs": [],
            }
        mode = r.get("mode", "jit")
        if mode == "jit":
            grouped[w]["jit_compile_ms"] += float(r.get("compile_ms", 0.0))
            execs = r.get("exec_ms", [])
            if isinstance(execs, (int, float)):
                execs = [float(execs)]
            if len(execs) > 1:
                grouped[w]["jit_execs"].extend([float(x) for x in execs[1:]])
            elif len(execs) == 1:
                grouped[w]["jit_execs"].extend([float(x) for x in execs])
        elif mode == "aot":
            grouped[w]["aot_load_ms"] += float(r.get("load_ms", 0.0))
            execs = r.get("exec_ms", [])
            if isinstance(execs, (int, float)):
                execs = [float(execs)]
            if len(execs) > 1:
                grouped[w]["aot_execs"].extend([float(x) for x in execs[1:]])
            elif len(execs) == 1:
                grouped[w]["aot_execs"].extend([float(x) for x in execs])

    out = {}
    for w, data in grouped.items():
        j_exec = sum(data["jit_execs"]) / len(data["jit_execs"]) if data["jit_execs"] else 0.0
        a_exec = sum(data["aot_execs"]) / len(data["aot_execs"]) if data["aot_execs"] else 0.0
        out[w] = {
            "jit_compile_ms": float(data["jit_compile_ms"]),
            "jit_exec_ms": float(j_exec),
            "aot_load_ms": float(data["aot_load_ms"]),
            "aot_exec_ms": float(a_exec),
        }
    return out
