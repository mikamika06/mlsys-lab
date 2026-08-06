def classify_readiness(logs, http_status, engine_state):
    """Classify pod state as process_up or engine_ready."""
    has_http_ok = http_status == 200
    is_engine_serving = engine_state.get("ready", False) is True and engine_state.get("graph_captured", False) is True
    has_ready_log = any("vLLM engine initialized and ready for inference" in log for log in logs)

    if has_http_ok and is_engine_serving and has_ready_log:
        return "engine_ready"
    return "process_up"


def parse_startup_phases(log_timestamps):
    """Parse runtime phases from log timestamp markers."""
    container_start = log_timestamps.get("container_start", 0.0)
    imports_loaded = log_timestamps.get("imports_loaded", container_start)
    weights_loaded = log_timestamps.get("weights_loaded", imports_loaded)
    compilation_done = log_timestamps.get("compilation_done", weights_loaded)
    engine_ready = log_timestamps.get("engine_ready", compilation_done)

    return {
        "process_bootstrap": max(0.0, imports_loaded - container_start),
        "weight_loading": max(0.0, weights_loaded - imports_loaded),
        "torch_compile": max(0.0, compilation_done - weights_loaded),
        "cudagraph_capture": max(0.0, engine_ready - compilation_done),
    }
