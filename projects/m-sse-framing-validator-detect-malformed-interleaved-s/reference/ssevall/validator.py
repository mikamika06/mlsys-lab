class FramingError(Exception):
    pass


def validate_sse_stream(raw_bytes: bytes) -> list[dict]:
    events = []
    blocks = raw_bytes.split(b"\n\n")
    for block in blocks:
        if not block.strip():
            continue
        lines = block.split(b"\n")
        event_data = {}
        for line in lines:
            if line.startswith(b":"):
                continue
            if b":" not in line:
                raise FramingError("Malformed line without colon delimiter")
            parts = line.split(b":", 1)
            field = parts[0].decode("utf-8").strip()
            value = parts[1].decode("utf-8").strip()
            if field == "data":
                event_data["data"] = value
            elif field == "event":
                event_data["event"] = value
        if "data" not in event_data:
            raise FramingError("Missing data field in SSE block")
        events.append(event_data)
    return events
