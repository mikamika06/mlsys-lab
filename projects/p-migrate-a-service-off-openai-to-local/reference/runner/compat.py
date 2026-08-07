def check_endpoint(name):
    supported = ["chat_completions", "embeddings", "streaming", "tool_calling"]
    return name in supported
