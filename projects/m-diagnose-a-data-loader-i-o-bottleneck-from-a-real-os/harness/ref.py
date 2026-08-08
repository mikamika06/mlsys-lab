import random

def get_test_cases():
    rng = random.Random(42)
    cases = []
    for i in range(5):
        events = [
            (0, "push", "step"),
            (10, "push", "load"),
            (20, "pop", "load"),
            (30, "pop", "step")
        ]
        osrt = [
            {"name": "read", "total_time_ms": float(rng.randint(1000, 5000)) + i * 100},
            {"name": "poll", "total_time_ms": float(rng.randint(10, 50))}
        ]
        cases.append((events, osrt))
    return cases
