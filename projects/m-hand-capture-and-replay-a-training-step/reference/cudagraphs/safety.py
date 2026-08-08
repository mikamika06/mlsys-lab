def classify_op_safety(op_type, inputs, outputs, memory_state):
    """Classifies CUDA graph safety into capture-safe, hard-error, or silently-wrong."""
    if op_type == "cpu_sync" or op_type == "cuda_malloc":
        return "hard-error"

    for inp in inputs:
        if memory_state.get(inp, {}).get("is_cpu_tensor", False):
            return "hard-error"

    for out in outputs:
        if memory_state.get(out, {}).get("is_aliased_to_input", False):
            return "silently-wrong"

    for inp in inputs:
        if memory_state.get(inp, {}).get("mutated_during_replay", False):
            return "silently-wrong"

    return "capture-safe"


def analyze_graph_safety(operations):
    results = []
    memory_state = {}

    for op in operations:
        op_type = op["op"]
        inputs = op.get("inputs", [])
        outputs = op.get("outputs", [])

        for inp in inputs:
            if inp not in memory_state:
                memory_state[inp] = op.get("input_states", {}).get(
                    inp,
                    {
                        "is_cpu_tensor": False,
                        "is_aliased_to_input": False,
                        "mutated_during_replay": False,
                    },
                )

        status = classify_op_safety(op_type, inputs, outputs, memory_state)

        for out in outputs:
            memory_state[out] = {
                "is_cpu_tensor": False,
                "is_aliased_to_input": op.get("creates_alias", False),
                "mutated_during_replay": op.get("mutates_input", False),
            }

        results.append({"op_id": op["id"], "safety": status})

    return results
