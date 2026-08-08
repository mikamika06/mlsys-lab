import numpy as np


def generate_trace_cases():
    cases = []
    for i in range(10):
        step_trace = [
            {"op": "add", "inputs": ["x", "y"], "output": "t1"},
            {"op": "relu", "inputs": ["t1"], "output": "t2"},
            {"op": "mul", "inputs": ["t2", "w"], "output": "out"},
        ]
        inputs_w = {"x": np.ones((4,), dtype=np.float64) * (i + 1),
                    "y": np.ones((4,), dtype=np.float64) * 2.0,
                    "w": np.ones((4,), dtype=np.float64) * 0.5}
        inputs_r = {"x": np.ones((4,), dtype=np.float64) * (i + 5),
                    "y": np.ones((4,), dtype=np.float64) * 3.0,
                    "w": np.ones((4,), dtype=np.float64) * 2.0}
        cases.append({"trace": step_trace, "warmup": inputs_w, "replay": inputs_r})
    return cases


def generate_safety_operations():
    return [
        {
            "id": 1,
            "op": "add",
            "inputs": ["a", "b"],
            "outputs": ["c"],
            "input_states": {
                "a": {"is_cpu_tensor": False, "is_aliased_to_input": False, "mutated_during_replay": False},
                "b": {"is_cpu_tensor": False, "is_aliased_to_input": False, "mutated_during_replay": False},
            },
            "creates_alias": False,
            "mutates_input": False,
        },
        {
            "id": 2,
            "op": "cpu_sync",
            "inputs": ["c"],
            "outputs": ["d"],
            "input_states": {
                "c": {"is_cpu_tensor": False, "is_aliased_to_input": False, "mutated_during_replay": False},
            },
            "creates_alias": False,
            "mutates_input": False,
        },
        {
            "id": 3,
            "op": "mul",
            "inputs": ["c", "e"],
            "outputs": ["f"],
            "input_states": {
                "c": {"is_cpu_tensor": False, "is_aliased_to_input": True, "mutated_during_replay": False},
                "e": {"is_cpu_tensor": False, "is_aliased_to_input": False, "mutated_during_replay": False},
            },
            "creates_alias": True,
            "mutates_input": False,
        },
        {
            "id": 4,
            "op": "relu",
            "inputs": ["f"],
            "outputs": ["g"],
            "input_states": {
                "f": {"is_cpu_tensor": False, "is_aliased_to_input": False, "mutated_during_replay": True},
            },
            "creates_alias": False,
            "mutates_input": True,
        },
    ]


def run_capture_ref(step_trace, warmup_inputs, replay_inputs):
    from cudagraphs.capture import GraphCaptureSimulator

    sim = GraphCaptureSimulator(step_trace)
    sim.warmup(warmup_inputs)
    sim.capture(warmup_inputs)
    return sim.replay(replay_inputs)


def analyze_safety_ref(ops):
    from cudagraphs.safety import analyze_graph_safety

    return analyze_graph_safety(ops)
