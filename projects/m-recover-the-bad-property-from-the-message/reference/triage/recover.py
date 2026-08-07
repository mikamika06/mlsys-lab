def parse_error_property(msg):
    parts = msg.split("property=")
    if len(parts) > 1:
        return {"property": parts[1].strip(), "raw": msg}
    return {"property": "unknown", "raw": msg}
