def map_openai_request_to_sampling_params(request_json):
    temperature = request_json.get("temperature", 1.0)
    max_tokens = request_json.get("max_tokens", 16)
    top_p = request_json.get("top_p", 1.0)
    extra_body = request_json.get("extra_body", {})

    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }
    if extra_body:
        params.update(extra_body)
    return params
