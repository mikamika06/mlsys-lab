"""Reference oracle definitions for grading harness."""

PIPELINE_NODES = [
    {"op_type": "Conv", "aligned_io": True, "has_control_flow": False},
    {"op_type": "Relu", "aligned_io": True, "has_control_flow": False},
    {"op_type": "CpuFallbackOp", "aligned_io": True, "has_control_flow": False},
    {"op_type": "MatMul", "aligned_io": False, "has_control_flow": False},
    {"op_type": "Reshape", "aligned_io": True, "dynamic_resizing": True},
    {"op_type": "Add", "aligned_io": True, "has_control_flow": False},
]

EXECUTION_STEPS = [
    {"type": "capture", "bindings": {"out_0": [10, 20], "out_1": [1, 1]}},
    {
        "type": "replay",
        "step_id": 1,
        "output_bindings": {"out_0": 0x1000, "out_1": 0x2000},
        "writes_output": True,
    },
    {
        "type": "replay",
        "step_id": 2,
        "output_bindings": {"out_0": 0x1000, "out_1": 0x2000},
        "writes_output": False,
    },
    {
        "type": "replay",
        "step_id": 3,
        "output_bindings": {"out_0": 0x1000, "out_1": 0x2000},
        "rebind_output": True,
    },
]

REPLAY_INPUTS = {
    1: {"out_0": [10, 20], "out_1": [2, 2]},
    2: {"out_0": [99, 99], "out_1": [3, 3]},
    3: {"out_0": [40, 50], "out_1": [4, 4]},
}

EXPECTED_STEP_OUTPUTS = {
    1: {"out_0": [10, 20], "out_1": [2, 2]},
    2: {"out_0": [99, 99], "out_1": [3, 3]},
    3: {"out_0": [40, 50], "out_1": [4, 4]},
}

ALLOCATIONS = [
    {"id": 1, "time": 1, "size": 1024},
    {"id": 2, "time": 2, "size": 2048},
    {"id": 3, "time": 4, "size": 512},
    {"id": 4, "time": 5, "size": 4096},
]

DEALLOCATIONS = [
    {"id": 1, "time": 3, "size": 1024},
    {"id": 3, "time": 6, "size": 512},
]

BLOCK_SIZE = 2048


def is_graph_legal(node_spec):
    op_type = node_spec.get("op_type", "")
    if op_type in ("CpuFallbackOp", "CustomHostOp", "DynamicShapeAlloc"):
        return False, "disallowed_op"
    if node_spec.get("has_control_flow", False):
        return False, "control_flow"
    if node_spec.get("allocates_host_pinned", False):
        return False, "host_pinned_alloc"
    if node_spec.get("dynamic_resizing", False):
        return False, "dynamic_shape"
    if not node_spec.get("aligned_io", True):
        return False, "unaligned_io"
    return True, "ok"


def classify_pipeline(nodes):
    legal_count = 0
    reasons = []
    for node in nodes:
        legal, reason = is_graph_legal(node)
        if legal:
            legal_count += 1
        else:
            reasons.append(reason)
    return {
        "is_legal": legal_count == len(nodes),
        "legal_nodes": legal_count,
        "total_nodes": len(nodes),
        "reasons": reasons,
    }


def simulate_capture_and_run(execution_steps, replay_inputs):
    captured_state = {}
    outputs_per_step = []
    for step in execution_steps:
        if step.get("type") == "capture":
            for k, v in step.get("bindings", {}).items():
                captured_state[k] = list(v)
        elif step.get("type") == "replay":
            step_inputs = replay_inputs.get(step["step_id"], {})
            out = {}
            for out_key, static_addr in step.get("output_bindings", {}).items():
                if step.get("rebind_output", False):
                    out[out_key] = list(step_inputs.get(out_key, []))
                else:
                    if out_key in step_inputs and step.get("writes_output", True):
                        captured_state[out_key] = list(step_inputs[out_key])
                    out[out_key] = list(captured_state.get(out_key, []))
            outputs_per_step.append({"step_id": step["step_id"], "outputs": out})
    return outputs_per_step


def detect_stale_outputs(capture_result):
    stale_steps = []
    for step_data in capture_result:
        step_id = step_data["step_id"]
        outputs = step_data["outputs"]
        is_stale = False
        for out_name, val in outputs.items():
            if step_data.get("expected", {}).get(out_name) is not None:
                if val != step_data["expected"][out_name]:
                    is_stale = True
                    break
            elif step_data.get("is_stale_flag", False):
                is_stale = True
                break
        if is_stale:
            stale_steps.append(step_id)
    return stale_steps


def analyze_arena_vs_rss(allocations, deallocations, block_size):
    active_bytes = 0
    arena_reserved_bytes = 0
    peak_active = 0
    peak_arena = 0
    timeline = []

    events = []
    for a in allocations:
        events.append((a["time"], "alloc", a["id"], a["size"]))
    for d in deallocations:
        events.append((d["time"], "dealloc", d["id"], d["size"]))
    events.sort(key=lambda x: (x[0], 0 if x[1] == "dealloc" else 1))

    for time, ev_type, alloc_id, size in events:
        if ev_type == "alloc":
            active_bytes += size
            if active_bytes > arena_reserved_bytes:
                needed = active_bytes - arena_reserved_bytes
                blocks = (needed + block_size - 1) // block_size
                arena_reserved_bytes += blocks * block_size
        else:
            active_bytes -= size

        peak_active = max(peak_active, active_bytes)
        peak_arena = max(peak_arena, arena_reserved_bytes)
        timeline.append({
            "time": time,
            "active_bytes": active_bytes,
            "arena_reserved_bytes": arena_reserved_bytes,
        })

    waste_at_peak = peak_arena - peak_active
    efficiency = peak_active / peak_arena if peak_arena > 0 else 1.0

    return {
        "peak_active_bytes": peak_active,
        "peak_arena_bytes": peak_arena,
        "waste_at_peak_bytes": waste_at_peak,
        "efficiency": efficiency,
        "timeline": timeline,
    }
