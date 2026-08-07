import numpy as np

def generate_case():
    np.random.seed(42)
    tokens = np.random.randn(16, 8)
    routing_map = np.random.randint(0, 2, size=(16, 4))
    return tokens, routing_map
