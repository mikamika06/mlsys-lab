import json


def parse_stream_metrics(ndjson_stream):
    lines = [line.strip() for line in ndjson_stream.split("\n") if line.strip()]
    events = [json.loads(line) for line in lines]

    start_time = None
    first_token_time = None
    last_token_time = None

    prompt_tokens = 0
    completion_tokens = 0

    for ev in events:
        t = ev["timestamp"]
        ev_type = ev["type"]
        if ev_type == "start":
            start_time = t
            prompt_tokens = ev.get("prompt_tokens", 0)
        elif ev_type == "token":
            if first_token_time is None:
                first_token_time = t
            last_token_time = t
            completion_tokens += ev.get("tokens", 1)
        elif ev_type == "end":
            if last_token_time is None:
                last_token_time = t

    ttft = first_token_time - start_time
    prefill_tok_per_sec = prompt_tokens / ttft if ttft > 0 else 0.0

    decode_duration = last_token_time - first_token_time
    if completion_tokens > 1 and decode_duration > 0:
        decode_tok_per_sec = (completion_tokens - 1) / decode_duration
    else:
        decode_tok_per_sec = 0.0

    return {
        "ttft": ttft,
        "prefill_tok_per_sec": prefill_tok_per_sec,
        "decode_tok_per_sec": decode_tok_per_sec,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
    }
