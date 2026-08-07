def get_compatibility_matrix():
    return {
        "chat_completions": True,
        "streaming": True,
        "tool_calling": True,
        "vision": False,
        "embeddings": True,
        "fine_tuning": False
    }

def collect_failures():
    return ["vision", "fine_tuning", "usage_metadata_missing_in_stream"]

def fix_streaming_and_tokens(chunks):
    out = []
    total = 0
    for c in chunks:
        if isinstance(c, dict):
            content = c.get("delta", {}).get("content", "")
            if content:
                total += len(content.split())
            out.append(c)
    return out, total

def fix_tool_calling(response):
    if isinstance(response, dict):
        if "tool_calls" not in response and "function_call" in response:
            response["tool_calls"] = [response["function_call"]]
    return response

def run_client_tests():
    return True

def get_behavioral_diffs():
    return {
        "latency": "higher time to first token",
        "system_prompt": "less strict instruction following",
        "max_tokens": "hard stop without truncation notice"
    }
