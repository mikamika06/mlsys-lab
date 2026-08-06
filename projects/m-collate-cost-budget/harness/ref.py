"""Reference dataset and collate benchmark scenario generators."""

import random


def make_deterministic_generator(seed=42):
    rnd = random.Random(seed)

    def sample_generator(batch_size):
        return [rnd.randint(10, 100) for _ in range(batch_size)]

    return sample_generator


def make_mock_collate_fn(base_cost_ms=0.5, per_item_cost_ms=0.2):
    def mock_collate(samples):
        batch_size = len(samples)
        total_time_sec = (base_cost_ms + per_item_cost_ms * batch_size) / 1000.0
        return 0.0, total_time_sec
    return mock_collate


def make_feature_collate_fns():
    def heavy_collate(samples):
        return 0.0, (2.0 + 1.0 * len(samples)) / 1000.0

    def medium_collate(samples):
        return 0.0, (1.0 + 0.4 * len(samples)) / 1000.0

    def light_collate(samples):
        return 0.0, (0.2 + 0.1 * len(samples)) / 1000.0

    return [
        {"name": "heavy", "fn": heavy_collate, "priority": 3},
        {"name": "medium", "fn": medium_collate, "priority": 2},
        {"name": "light", "fn": light_collate, "priority": 1},
    ]
