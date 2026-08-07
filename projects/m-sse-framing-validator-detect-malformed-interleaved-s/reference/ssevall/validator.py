def validate_sse_stream(stream_chunks):
    active_ids = set()
    malformed_detected = False
    interleaved_detected = False

    for chunk in stream_chunks:
        if not chunk.endswith("\n\n"):
            malformed_detected = True
        lines = chunk.strip().split("\n")
        stream_id = None
        has_data = False
        for line in lines:
            if line.startswith("id:"):
                stream_id = line[3:].strip()
            elif line.startswith("data:"):
                has_data = True
        if stream_id:
            if stream_id in active_ids:
                interleaved_detected = True
            active_ids.add(stream_id)

    return {
        "malformed": malformed_detected,
        "interleaved": interleaved_detected,
        "valid": not malformed_detected and not interleaved_detected
    }
