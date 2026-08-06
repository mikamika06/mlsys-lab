import sys
sys.path.insert(0, ".")

from triton_remapping.remapping import remap_pid, generate_grid_schedule
from triton_remapping.cache_sim import simulate_block_loads
from triton_remapping.search import find_optimal_group_size


def test_remap_pid_coverage():
    num_pid_m, num_pid_n = 10, 8
    group_size_m = 4
    schedule = generate_grid_schedule(num_pid_m, num_pid_n, group_size_m)

    assert len(schedule) == num_pid_m * num_pid_n
    assert len(set(schedule)) == num_pid_m * num_pid_n

    for m, n in schedule:
        assert 0 <= m < num_pid_m
        assert 0 <= n < num_pid_n


def test_grouped_loads_lower_than_linear():
    num_pid_m, num_pid_n = 32, 32
    cap = 16
    linear_loads = simulate_block_loads(num_pid_m, num_pid_n, 1, cap)
    grouped_loads = simulate_block_loads(num_pid_m, num_pid_n, 8, cap)
    assert grouped_loads < linear_loads


def test_optimal_search_validity():
    num_pid_m, num_pid_n = 16, 16
    cap = 8
    best_g = find_optimal_group_size(num_pid_m, num_pid_n, cap, max_group_size=8)
    assert 1 <= best_g <= 8
