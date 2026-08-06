import math


def generate_test_cases():
    return [
        {
            "logs": ["vLLM engine initialized and ready for inference"],
            "http_status": 200,
            "engine_state": {"ready": True, "graph_captured": True},
            "expected_readiness": "engine_ready",
        },
        {
            "logs": ["vLLM engine initialized and ready for inference"],
            "http_status": 200,
            "engine_state": {"ready": True, "graph_captured": False},
            "expected_readiness": "process_up",
        },
        {
            "logs": ["Loading model weights..."],
            "http_status": 200,
            "engine_state": {"ready": False, "graph_captured": False},
            "expected_readiness": "process_up",
        },
        {
            "logs": ["vLLM engine initialized and ready for inference"],
            "http_status": 503,
            "engine_state": {"ready": True, "graph_captured": True},
            "expected_readiness": "process_up",
        },
    ]


def generate_cooldown_cases():
    return [
        {
            "timestamps": {
                "container_start": 0.0,
                "imports_loaded": 5.2,
                "weights_loaded": 25.4,
                "compilation_done": 85.4,
                "engine_ready": 105.4,
            },
            "warm_cache": True,
            "speedup": 5.0,
            "margin_pct": 15.0,
        },
        {
            "timestamps": {
                "container_start": 100.0,
                "imports_loaded": 112.0,
                "weights_loaded": 142.0,
                "compilation_done": 322.0,
                "engine_ready": 352.0,
            },
            "warm_cache": False,
            "speedup": 3.0,
            "margin_pct": 10.0,
        },
        {
            "timestamps": {
                "container_start": 10.0,
                "imports_loaded": 18.0,
                "weights_loaded": 48.0,
                "compilation_done": 168.0,
                "engine_ready": 188.0,
            },
            "warm_cache": True,
            "speedup": 4.0,
            "margin_pct": 25.0,
        },
    ]


def ref_classify_readiness(logs, http_status, engine_state):
    has_http_ok = http_status == 200
    is_engine_serving = engine_state.get("ready", False) is True and engine_state.get("graph_captured", False) is True
    has_ready_log = any("vLLM engine initialized and ready for inference" in log for log in logs)

    if has_http_ok and is_engine_serving and has_ready_log:
        return "engine_ready"
    return "process_up"


def ref_parse_startup_phases(log_timestamps):
    c_start = log_timestamps.get("container_start", 0.0)
    i_loaded = log_timestamps.get("imports_loaded", c_start)
    w_loaded = log_timestamps.get("weights_loaded", i_loaded)
    c_done = log_timestamps.get("compilation_done", w_loaded)
    e_ready = log_timestamps.get("engine_ready", c_done)

    return {
        "process_bootstrap": max(0.0, i_loaded - c_start),
        "weight_loading": max(0.0, w_loaded - i_loaded),
        "torch_compile": max(0.0, c_done - w_loaded),
        "cudagraph_capture": max(0.0, e_ready - c_done),
    }


def ref_compute_cooldown(phase_times, warm_compile_cache, cache_speedup_factor, safety_margin_pct):
    compile_time = phase_times.get("torch_compile", 0.0)
    if warm_compile_cache:
        compile_time = compile_time / cache_speedup_factor

    total = (
        phase_times.get("process_bootstrap", 0.0)
        + phase_times.get("weight_loading", 0.0)
        + compile_time
        + phase_times.get("cudagraph_capture", 0.0)
    )
    return int(math.ceil(total * (1.0 + (safety_margin_pct / 100.0))))
