import random

def get_test_cases():
    rng = random.Random(42)
    cases = []
    for _ in range(5):
        shapes = []
        for _ in range(30):
            b = rng.choice([1, 2, 4, 8])
            s = rng.choice([128, 256, 512])
            shapes.append((b, s))
        cases.append(shapes)
    return cases
