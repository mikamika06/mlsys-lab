def track_model(state, model_name, memory_bytes, timestamp):
    if model_name in state:
        state[model_name]["last_used"] = timestamp
        state[model_name]["access_count"] += 1
    else:
        state[model_name] = {
            "memory_bytes": memory_bytes,
            "last_used": timestamp,
            "access_count": 1,
            "loaded": True
        }
    return state
