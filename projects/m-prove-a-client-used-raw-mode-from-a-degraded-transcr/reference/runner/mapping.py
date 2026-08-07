def map_openai_to_native(request):
    native = {
        "prompt": "",
        "temperature": request.get("temperature", 0.7),
        "max_tokens": request.get("max_tokens", 128),
        "stream": request.get("stream", False)
    }
    messages = request.get("messages", [])
    prompt_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        prompt_parts.append(f"{role}: {content}")
    native["prompt"] = "\n".join(prompt_parts)
    return native
