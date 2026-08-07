def sample_matrix():
    return {
        "chat_completions": True,
        "streaming": True,
        "tool_calling": True,
        "vision": False,
        "embeddings": True,
        "fine_tuning": False
    }

def sample_chunks():
    return [{"delta": {"content": "test streaming chunk"}}]
