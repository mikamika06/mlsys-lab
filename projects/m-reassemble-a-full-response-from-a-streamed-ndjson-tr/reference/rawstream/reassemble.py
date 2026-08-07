import json

def reassemble_stream(lines):
    chunks = []
    for line in lines:
        if not line or not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "response" in data:
            chunks.append(data["response"])
        elif "text" in data:
            chunks.append(data["text"])
        elif "token" in data:
            chunks.append(data["token"])
        elif "choices" in data:
            for choice in data["choices"]:
                if "text" in choice:
                    chunks.append(choice["text"])
    return "".join(chunks)
