import random


def generate_fixtures():
    random.seed(42)
    modes = ["default", "reduce-overhead", "max-autotune"]
    fixtures = []
    for _ in range(10):
        base_config = {
            "triton.cudagraphs": False,
            "max_autotune": False,
            "epilogue_fusion": True,
        }
        target_mode = random.choice(modes)
        fixtures.append((base_config, target_mode))
    return fixtures


def generate_breakeven_cases():
    random.seed(1337)
    cases = []
    for _ in range(20):
        compile_overhead = random.uniform(5.0, 120.0)
        t_default = random.uniform(0.01, 0.5)
        t_autotune = t_default * random.uniform(0.6, 0.95)
        cases.append((compile_overhead, t_default, t_autotune))
    return cases


def generate_budget_cases():
    random.seed(9001)
    cases = []
    for _ in range(15):
        overhead = random.uniform(10.0, 50.0)
        t_def = 0.1
        t_rov = 0.09
        t_max = 0.07
        budget_steps = random.randint(100, 2000)
        cases.append((overhead, t_def, t_rov, t_max, budget_steps))
    return cases
