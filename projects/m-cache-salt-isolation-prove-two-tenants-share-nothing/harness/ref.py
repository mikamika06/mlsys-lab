import random


def get_test_cases():
    rnd = random.Random(42)
    cases = []
    for i in range(5):
        blocks_a = [rnd.randint(1, 100) for _ in range(10)]
        blocks_b = [rnd.randint(1, 100) for _ in range(10)]
        salt_a = f"salt_a_{i}"
        salt_b = f"salt_b_{i}"
        cases.append((blocks_a, blocks_b, salt_a, salt_b))
    return cases


def get_surgery_cases():
    return [
        ([10, 20, 999, 30, 40], 999),
        ([999, 1, 2, 3], 999),
        ([1, 2, 3, 4], 999)
    ]


def get_lora_cases():
    return [
        ("block_content_1", "lora_42", "salt_x"),
        ("block_content_2", "lora_99", "salt_y")
    ]
