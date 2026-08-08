def map_openai_request(req_json: dict) -> dict:
    sampling_params = {}
    if "temperature" in req_json:
        sampling_params["temperature"] = float(req_json["temperature"])
    if "top_p" in req_json:
        sampling_params["top_p"] = float(req_json["top_p"])
    if "max_tokens" in req_json:
        sampling_params["max_tokens"] = int(req_json["max_tokens"])
    if "stop" in req_json:
        stop = req_json["stop"]
        if isinstance(stop, str):
            sampling_params["stop"] = [stop]
        elif isinstance(stop, list):
            sampling_params["stop"] = list(stop)
    if "extra_body" in req_json and isinstance(req_json["extra_body"], dict):
        for k, v in req_json["extra_body"].items():
            sampling_params[k] = v
    return sampling_params
