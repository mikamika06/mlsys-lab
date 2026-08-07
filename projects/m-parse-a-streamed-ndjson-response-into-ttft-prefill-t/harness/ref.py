import json
import random
import numpy as np


def generate_ndjson_stream(prompt_tokens, completion_tokens, prefill_speed, decode_speed, start_t=1000.0):
    ttft = prompt_tokens / prefill_speed
    lines = []
    lines.append(json.dumps({"type": "start", "timestamp": start_t, "prompt_tokens": prompt_tokens}))

    curr_t = start_t + ttft
    for i in range(completion_tokens):
        lines.append(json.dumps({"type": "token", "timestamp": curr_t, "tokens": 1}))
        if i < completion_tokens - 1:
            curr_t += 1.0 / decode_speed

    lines.append(json.dumps({"type": "end", "timestamp": curr_t}))
    return "\n".join(lines)


def generate_dataset(num_traces=5, seed=42):
    rng = random.Random(seed)
    dataset = []
    for _ in range(num_traces):
        p_toks = rng.randint(200, 2000)
        c_toks = rng.randint(20, 200)
        p_speed = rng.uniform(400.0, 1200.0)
        d_speed = rng.uniform(20.0, 80.0)
        stream = generate_ndjson_stream(p_toks, c_toks, p_speed, d_speed)
        dataset.append({
            "stream": stream,
            "prompt_tokens": p_toks,
            "completion_tokens": c_toks,
            "expected_p_speed": p_speed,
            "expected_d_speed": d_speed
        })
    return dataset


def ref_parse_stream_metrics(ndjson_stream):
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
    decode_tok_per_sec = (completion_tokens - 1) / decode_duration if completion_tokens > 1 and decode_duration > 0 else 0.0

    return {
        "ttft": ttft,
        "prefill_tok_per_sec": prefill_tok_per_sec,
        "decode_tok_per_sec": decode_tok_per_sec,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
    }


def ref_compute_median_decode_throughput(runs_ndjson_streams, warmup_runs=1):
    valid = runs_ndjson_streams[warmup_runs:]
    if not valid:
        return 0.0
    rates = [ref_parse_stream_metrics(s)["decode_tok_per_sec"] for s in valid]
    return float(np.median(rates))


def ref_classify_workload_dominance(ttft, prefill_tok_per_sec, decode_tok_per_sec, prompt_tokens, completion_tokens):
    prefill_time = ttft
    decode_time = (completion_tokens - 1) / decode_tok_per_sec if decode_tok_per_sec > 0 else 0.0
    return "prefill-dominated" if prefill_time >= decode_time else "decode-dominated"
