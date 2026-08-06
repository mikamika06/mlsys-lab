import scalezero.optimizer as opt

def test_finds_smallest_timeout():
    traffic = [0, 0, 10, 0, 0, 10]
    ans = opt.find_optimal_timeout(traffic, 1, 0.5)
    assert ans == 3, f"Expected 3, got {ans}"
