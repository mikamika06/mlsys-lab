import random
from app.components import MockModel, MockFSM
from app.decoder import generate_safe

def run_1000_times() -> list[list[int]]:
    results = []
    rng = random.Random(42)
    for i in range(1000):
        model = MockModel(i)
        fsm = MockFSM()
        budget = rng.randint(5, 15)
        tokens = generate_safe(model, fsm, budget)
        results.append(tokens)
    return results
