def validate_openai_payload(payload):
    required_keys = ["id", "object", "created", "model", "choices", "usage"]
    for k in required_keys:
        if k not in payload:
            return False
    if not isinstance(payload["choices"], list) or len(payload["choices"]) == 0:
        return False
    choice = payload["choices"][0]
    if "message" not in choice or "content" not in choice["message"]:
        return False
    return True
