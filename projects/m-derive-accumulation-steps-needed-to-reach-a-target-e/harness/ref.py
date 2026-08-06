import random

def get_test_cases():
    random.seed(42)
    cases = []
    for _ in range(5):
        devices = random.choice([1, 2, 4, 8])
        per_device = random.choice([1, 2, 4])
        multiplier = random.choice([1, 2, 4, 8, 16])
        target = devices * per_device * multiplier
        cases.append({
            "target_effective_batch_size": target,
            "per_device_batch_size": per_device,
            "num_devices": devices,
            "expected_steps": multiplier
        })
    return cases
