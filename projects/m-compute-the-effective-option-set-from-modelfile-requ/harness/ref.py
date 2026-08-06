import random

def get_test_cases():
    random.seed(42)
    cases = []
    for i in range(5):
        mf = {"temperature": 0.5 + i * 0.1, "num_ctx": 2048 + i * 512}
        req = {"temperature": 0.1 * i}
        env = {"LLAMA_TEMP": str(0.3 + i * 0.05)}
        cases.append((mf, req, env))
    return cases
