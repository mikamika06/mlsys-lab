def continue_conversation(context, messages):
    history = list(context) if context else []
    for m in messages:
        history.append({
            "role": m.get("role", "user"),
            "content": m.get("content", ""),
            "state_carried": True
        })
    return history
