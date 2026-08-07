def get_compatibility_matrix():
    return {
        "chat_completions": True,
        "embeddings": True,
        "fine_tuning": False,
        "images": False
    }

def switch_base_url(client, url):
    client.base_url = url
    return client.base_url

def collect_errors(client, payloads):
    errors = []
    for p in payloads:
        if not isinstance(p, dict) or "model" not in p:
            errors.append("missing_model")
        else:
            errors.append(None)
    return errors
