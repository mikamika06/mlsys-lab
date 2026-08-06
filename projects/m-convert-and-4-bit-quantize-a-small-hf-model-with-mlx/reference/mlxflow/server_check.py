def validate_openai_schema(payload):
    if not isinstance(payload, dict):
        return False
    if payload.get("object") != "chat.completion":
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            return False
        message = choice.get("message")
        if not isinstance(message, dict):
            return False
        if message.get("role") not in ("assistant", "user", "system"):
            return False
        if not isinstance(message.get("content"), str):
            return False
    return True
