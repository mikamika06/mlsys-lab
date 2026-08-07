import json

def parse_stream(chunks, schema):
    valid_chunks = []
    for chunk in chunks:
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        valid = True
        for req in schema.get("required", []):
            if req not in data:
                valid = False
                break
        if valid:
            valid_chunks.append(chunk)
    return valid_chunks
