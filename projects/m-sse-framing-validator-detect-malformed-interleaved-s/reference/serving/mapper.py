def map_request_to_sampling_params(request_json):
    if not isinstance(request_json, dict):
        return {}
    params = {}
    if "temperature" in request_json:
        params["temperature"] = request_json["temperature"]
    if "top_p" in request_json:
        params["top_p"] = request_json["top_p"]
    if "max_tokens" in request_json:
        params["max_tokens"] = request_json["max_tokens"]
    if "stop" in request_json:
        params["stop"] = request_json["stop"]
    if "extra_body" in request_json and isinstance(request_json["extra_body"], dict):
        for k, v in request_json["extra_body"].items():
            params[k] = v
    return params
