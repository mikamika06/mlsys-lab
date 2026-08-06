import random


def generate_skew_test_cases():
    rng = random.Random(42)
    versions = ["0.1.20", "0.1.25", "0.1.30", "0.2.0"]
    cases = []
    for _ in range(10):
        v1 = rng.choice(versions)
        v2 = rng.choice(versions)
        cases.append((v1, {"version": v2}))
    return cases


def generate_binding_test_cases():
    return [
        ("http://127.0.0.1:11434", [{"host": "127.0.0.1", "port": 11434}], True),
        ("0.0.0.0:11434", [{"host": "0.0.0.0", "port": 11434}], True),
        ("192.168.1.50:11434", [{"host": "127.0.0.1", "port": 11434}], False),
        ("http://localhost:11434", [{"host": "127.0.0.1", "port": 11434}], True),
    ]


def mock_request_runner():
    call_count = [0]

    def req():
        call_count[0] += 1
        if call_count[0] == 1:
            return (150.0, "ok")
        return (10.0, "ok")

    return req
