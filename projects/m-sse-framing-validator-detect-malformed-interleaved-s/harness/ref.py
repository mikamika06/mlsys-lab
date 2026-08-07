STREAMS = [
    (["id: a\ndata: foo\n\n", "id: b\ndata: bar\n\n"], {"malformed": False, "interleaved": False, "valid": True}),
    (["id: a\ndata: foo\n", "id: b\ndata: bar\n\n"], {"malformed": True, "interleaved": False, "valid": False}),
    (["id: a\ndata: foo\n\n", "id: a\ndata: bar\n\n"], {"malformed": False, "interleaved": True, "valid": False}),
]

REQUESTS = [
    ({"temperature": 0.7, "max_tokens": 100}, {"temperature": 0.7, "max_tokens": 100, "top_p": 1.0}),
    ({"temperature": 0.2, "extra_body": {"repetition_penalty": 1.1}}, {"temperature": 0.2, "max_tokens": 16, "top_p": 1.0, "repetition_penalty": 1.1}),
]

DIVERGENCES = [
    ("<|im_start|>user\nHi<|im_end|>", "Hi"),
    ("<|system|>Prompt<|end|>Hello", "Hello"),
]

def validate_sse_stream(stream_chunks):
    active_ids = set()
    malformed_detected = False
    interleaved_detected = False
    for chunk in stream_chunks:
        if not chunk.endswith("\n\n"):
            malformed_detected = True
        lines = chunk.strip().split("\n")
        stream_id = None
        for line in lines:
            if line.startswith("id:"):
                stream_id = line[3:].strip()
        if stream_id:
            if stream_id in active_ids:
                interleaved_detected = True
            active_ids.add(stream_id)
    return {
        "malformed": malformed_detected,
        "interleaved": interleaved_detected,
        "valid": not malformed_detected and not interleaved_detected
    }

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

def locate_divergence_tokens(chat_prompt, completion_prompt):
    chat_tokens = chat_prompt.split()
    comp_tokens = completion_prompt.split()
    divergence_index = 0
    for i, (c, p) in enumerate(zip(chat_tokens, comp_tokens)):
        if c != p:
            divergence_index = i
            break
    else:
        divergence_index = min(len(chat_tokens), len(comp_tokens))
    return {
        "divergence_index": divergence_index,
        "chat_template_prefix": chat_tokens[:divergence_index],
        "completion_prefix": comp_tokens[:divergence_index]
    }
