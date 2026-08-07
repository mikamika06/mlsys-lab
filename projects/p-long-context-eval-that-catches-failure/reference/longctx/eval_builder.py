import numpy as np

def generate_dataset(context_lengths, positions, num_samples=10):
    dataset = []
    for length in context_lengths:
        for pos in positions:
            for i in range(num_samples):
                dataset.append({
                    "context_length": length,
                    "relative_position": pos,
                    "fact_id": f"fact_{length}_{pos}_{i}",
                    "text": f"Context start... fact_{length}_{pos}_{i} ... context end."
                })
    return dataset
