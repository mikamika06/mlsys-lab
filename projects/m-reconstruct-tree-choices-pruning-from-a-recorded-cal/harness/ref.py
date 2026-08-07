import numpy as np

def generate_fixtures():
    np.random.seed(42)
    fixtures = []
    for _ in range(3):
        probs = np.random.uniform(0.01, 1.0, size=(4, 5))
        probs = probs / probs.sum(axis=1, keepdims=True)
        fixtures.append(probs)
    return fixtures

FIXTURES = generate_fixtures()

def reconstruct_tree_choices(fixture):
    choices = []
    for row in fixture:
        sorted_idx = np.argsort(row)[::-1]
        choices.append(sorted_idx[:3].tolist())
    return choices

def evaluate_budget(medusa_params, eagle_params):
    return int(medusa_params == eagle_params)
