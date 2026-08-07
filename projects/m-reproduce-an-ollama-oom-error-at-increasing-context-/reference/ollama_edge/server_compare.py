def normalize_ollama_response(payload):
    if "response" in payload:
        text = payload["response"]
    elif "choices" in payload and len(payload["choices"]) > 0:
        text = payload["choices"][0].get("text", "") or payload["choices"][0].get("message", {}).get("content", "")
    else:
        text = ""
    tokens = payload.get("eval_count", payload.get("usage", {}).get("completion_tokens", 0))
    return {"text": text, "tokens": tokens}


def normalize_mlx_response(payload):
    text = ""
    if "choices" in payload and len(payload["choices"]) > 0:
        text = payload["choices"][0].get("text", "") or payload["choices"][0].get("message", {}).get("content", "")
    tokens = payload.get("usage", {}).get("completion_tokens", 0)
    return {"text": text, "tokens": tokens}


def compare_responses(ollama_payload, mlx_payload):
    o = normalize_ollama_response(ollama_payload)
    m = normalize_mlx_response(mlx_payload)
    return o["text"] == m["text"] and o["tokens"] == m["tokens"]
