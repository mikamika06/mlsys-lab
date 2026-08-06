from triton_calc.regs import effective_regs
from triton_calc.occupancy import max_resident_blocks


def test_rounding_invariant():
    assert effective_regs(31, 8) == 32
    assert effective_regs(32, 8) == 32
    assert effective_regs(33, 8) == 40


def test_occupancy_bounds():
    spec = {
        "max_regs_per_sm": 65536,
        "max_threads_per_sm": 1536,
        "max_blocks_per_sm": 16,
        "reg_granularity": 8,
    }
    blocks = max_resident_blocks(64, 256, spec)
    assert blocks > 0
    assert blocks <= 16
