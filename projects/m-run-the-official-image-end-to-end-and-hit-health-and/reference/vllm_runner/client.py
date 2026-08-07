def parse_health(response_status, response_body):
    if response_status == 200 and response_body.strip() in ("", "OK", "healthy"):
        return {"status": "healthy", "ready": True}
    return {"status": "unhealthy", "ready": False}

def parse_completion(payload):
    if not isinstance(payload, dict):
        raise ValueError("payload must be dict")
    choices = payload.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("text", "")
