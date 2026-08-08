def resolve_effective_dtype(states, op_name):
    for s in states:
        if s["id"] == op_name:
            if not s["effective_enabled"]:
                return "float32"
            return s["effective_dtype"]
    return "float32"
