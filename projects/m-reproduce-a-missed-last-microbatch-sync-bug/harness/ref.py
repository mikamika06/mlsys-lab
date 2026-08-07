from syncbug.sim import simulate_accumulation
from syncbug.bench import reduce_time_ratio
from syncbug.breakeven import compute_breakeven_k

TEST_CASES_SIM = [1, 2, 4, 8]
TEST_CASES_BENCH = [(10000, 4, 2), (1000000, 4, 2)]
TEST_CASES_BREAKEVEN = [(2, 1e-8, 1e10), (4, 2e-8, 5e9)]
