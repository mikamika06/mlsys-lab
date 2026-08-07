"""CUDA graph capture runner and output freshness validator implementation."""


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
