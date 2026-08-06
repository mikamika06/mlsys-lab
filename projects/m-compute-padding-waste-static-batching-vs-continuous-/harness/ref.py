import random


def generate_workload(num_requests=20, seed=42):
    rng = random.Random(seed)
    requests = []
    for i in range(num_requests):
        arrival = rng.randint(0, 10) * 10
        p_len = rng.randint(5, 50)
        d_len = rng.randint(5, 30)
        requests.append({
            "id": i + 1,
            "arrival_time": arrival,
            "prompt_len": p_len,
            "decode_len": d_len,
        })
    return requests


def generate_logs(seed=42):
    rng = random.Random(seed)
    logs = []
    for mode in ["static", "continuous"]:
        for _ in range(5):
            tokens = rng.randint(1000, 5000)
            mult = 2.5 if mode == "static" else 1.0
            exec_time = float(tokens) / (100.0 / mult)
            logs.append({
                "mode": mode,
                "total_useful_tokens": tokens,
                "execution_time_sec": exec_time,
            })
    return logs
