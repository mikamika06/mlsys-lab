def patch_system_messages(messages):
    if not messages:
        return []
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]
    if not system_msgs:
        return other_msgs
    combined_content = "\n\n".join(m.get("content", "") for m in system_msgs)
    new_system = {"role": "system", "content": combined_content}
    return [new_system] + other_msgs
