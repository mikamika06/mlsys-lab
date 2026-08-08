def parse_request(raw):
    arrival = raw["arrival_time"]
    start = raw["start_time"]
    finish = raw["finish_time"]
    prompt_tokens = raw["prompt_tokens"]
    output_tokens = raw["output_tokens"]
    queue_time = start - arrival
    prefill_time = raw.get("prefill_time", prompt_tokens * 0.5)
    output_time = finish - (start + prefill_time)
    total_latency = finish - arrival
    return {
        "request_id": raw["request_id"],
        "queue_time": queue_time,
        "prefill_time": prefill_time,
        "output_time": output_time,
        "total_latency": total_latency,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens
    }
