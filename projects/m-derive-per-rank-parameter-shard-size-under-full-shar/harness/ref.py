import random

def get_test_cases():
    random.seed(42)
    cases = []
    for _ in range(20):
        total_params = random.randint(100, 100000)
        world_size = random.randint(2, 16)
        cases.append((total_params, world_size))
    return cases

def get_comm_cases():
    random.seed(43)
    cases = []
    strategies = ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"]
    for _ in range(30):
        num_params = random.randint(1000, 50000)
        bytes_per_param = random.choice([2, 4])
        strategy = random.choice(strategies)
        world_size = random.randint(2, 8)
        cases.append((num_params, bytes_per_param, strategy, world_size))
    return cases

def get_memory_cases():
    random.seed(44)
    cases = []
    for _ in range(20):
        param_bytes = random.randint(10000, 1000000)
        input_bytes = random.randint(1000, 100000)
        world_size = random.randint(2, 8)
        cases.append((param_bytes, input_bytes, world_size))
    return cases
