def fix_streaming(stream):
    chunks = []
    for chunk in stream:
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                chunks.append(delta.content)
    return "".join(chunks)

def count_tokens(response):
    if hasattr(response, "usage") and response.usage:
        return response.usage.total_tokens
    return 0
