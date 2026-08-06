import random

def generate_cases():
    random.seed(42)
    logs_cases = []
    for i in range(5):
        lines = [
            f"llama_print_timings: prompt eval time = 10.00 ms / 10 tokens",
            f"slot {i}: shift context: shifted 32 tokens, kept 128",
            f"warning: context truncated due to overflow in slot {i}",
            f"slot {i}: decoded 16 tokens in 50.00 ms"
        ]
        logs_cases.append((lines, {"slot": i, "shifted_tokens": 32, "kept_tokens": 128, "truncated": True}))

    perf_cases = []
    for i in range(5):
        seq = [100.0 + random.randint(1, 10) for _ in range(5)]
        batch = [120.0 + random.randint(1, 10) for _ in range(5)]
        ratio = sum(batch) / sum(seq)
        perf_cases.append((seq, batch, ratio))

    payload_cases = []
    for i in range(5):
        payload = {
            "id": f"chatcmpl-{i}",
            "object": "chat.completion",
            "created": 1700000000 + i,
            "model": "gguf-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"response {i}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        payload_cases.append((payload, True))
    return logs_cases, perf_cases, payload_cases
