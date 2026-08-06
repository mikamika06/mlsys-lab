import math

SPECS = [
    {
        "max_regs_per_sm": 65536,
        "max_threads_per_sm": 1536,
        "max_blocks_per_sm": 16,
        "reg_granularity": 8,
    },
    {
        "max_regs_per_sm": 65536,
        "max_threads_per_sm": 2048,
        "max_blocks_per_sm": 32,
        "reg_granularity": 8,
    },
    {
        "max_regs_per_sm": 32768,
        "max_threads_per_sm": 1024,
        "max_blocks_per_sm": 16,
        "reg_granularity": 4,
    },
]

TEST_CASES = [
    {"regs_per_thread": 31, "threads_per_block": 256, "spec": SPECS[0]},
    {"regs_per_thread": 64, "threads_per_block": 512, "spec": SPECS[1]},
    {"regs_per_thread": 128, "threads_per_block": 128, "spec": SPECS[2]},
    {"regs_per_thread": 25, "threads_per_block": 256, "spec": SPECS[0]},
    {"regs_per_thread": 45, "threads_per_block": 1024, "spec": SPECS[1]},
]


def compute_effective_regs(regs_per_thread, reg_granularity):
    return math.ceil(regs_per_thread / reg_granularity) * reg_granularity


def compute_max_blocks(regs_per_thread, threads_per_block, spec):
    eff_regs = compute_effective_regs(regs_per_thread, spec["reg_granularity"])
    regs_per_block = eff_regs * threads_per_block
    if regs_per_block == 0:
        blocks_by_regs = spec["max_blocks_per_sm"]
    else:
        blocks_by_regs = spec["max_regs_per_sm"] // regs_per_block
    blocks_by_threads = spec["max_threads_per_sm"] // threads_per_block
    return min(blocks_by_regs, blocks_by_threads, spec["max_blocks_per_sm"])


def evaluate_budget(test_cases):
    results = []
    for tc in test_cases:
        blocks = compute_max_blocks(
            tc["regs_per_thread"], tc["threads_per_block"], tc["spec"]
        )
        results.append(
            {
                "regs": tc["regs_per_thread"],
                "threads": tc["threads_per_block"],
                "max_blocks": blocks,
            }
        )
    return results
