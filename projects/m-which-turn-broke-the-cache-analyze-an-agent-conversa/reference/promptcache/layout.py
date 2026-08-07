def format_optimized_prompt(system_prompt, history, user_message, dynamic_state):
    parts = [system_prompt]
    for u, a in history:
        parts.append(f"User: {u}")
        parts.append(f"Assistant: {a}")
    parts.append(f"User: {user_message}")
    if dynamic_state:
        parts.append(f"State: {dynamic_state}")
    return "\n".join(parts)
