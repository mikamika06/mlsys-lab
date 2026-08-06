import numpy as np

def generate_admission_test_data():
    np.random.seed(42)
    requests = []
    for i in range(15):
        requests.append({
            "id": i + 1,
            "prompt_len": int(np.random.randint(10, 100)),
            "remaining_output": int(np.random.randint(2, 5))
        })
    return requests

def generate_latency_test_data():
    np.random.seed(123)
    num_reqs = 30
    requests = []
    arrival_times = []
    curr = 0.0
    for i in range(num_reqs):
        curr += float(np.random.exponential(0.01))
        arrival_times.append(curr)
        is_long = (i % 7 == 0)
        requests.append({
            "id": i + 1,
            "prompt_len": 500 if is_long else 20,
            "output_len": 5 if is_long else 10
        })
    return requests, arrival_times
