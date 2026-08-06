"""OpenAI SDK payload transformation adapter."""


def transform_request(shape_name, payload):
    out = dict(payload)
    out["_shape"] = shape_name
    out["adapter_version"] = "1.0"
    
    if shape_name == "stream_with_usage":
        out["stream"] = True
        if "stream_options" not in out:
            out["stream_options"] = {"include_usage": True}
    elif shape_name == "json_schema_format":
        if "response_format" not in out:
            out["response_format"] = {"type": "json_object"}
    elif shape_name == "tool_calls":
        if "tools" not in out:
            out["tools"] = []
    elif shape_name == "multimodal_content":
        out["is_multimodal"] = True
    elif shape_name == "logprobs_topk":
        out["logprobs"] = True
    elif shape_name == "multi_choice_n":
        out["n"] = payload.get("n", 1)
    elif shape_name == "token_array_prompt":
        out["is_token_ids"] = True
    elif shape_name == "stop_sequences":
        if isinstance(out.get("stop"), str):
            out["stop"] = [out["stop"]]
            
    return out


def validate_response(shape_name, response):
    if not isinstance(response, dict):
        return False
    if "error" in response:
        return False
    if "choices" not in response or not isinstance(response["choices"], list):
        return False
    if len(response["choices"]) == 0:
        return False

    if shape_name == "stream_with_usage":
        return "usage" in response
    elif shape_name == "json_schema_format":
        return "content" in response["choices"][0].get("message", {})
    elif shape_name == "tool_calls":
        return "tool_calls" in response["choices"][0].get("message", {})
    elif shape_name == "multimodal_content":
        return "content" in response["choices"][0].get("message", {})
    elif shape_name == "logprobs_topk":
        return "logprobs" in response["choices"][0]
    elif shape_name == "multi_choice_n":
        return len(response["choices"]) >= 1
    elif shape_name == "token_array_prompt":
        return "text" in response["choices"][0]
    elif shape_name == "stop_sequences":
        return response["choices"][0].get("finish_reason") == "stop" or "content" in response["choices"][0].get("message", {})

    return True
