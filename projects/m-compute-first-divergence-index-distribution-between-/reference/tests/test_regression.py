from divergence.analyze import compute_divergences

def test_divergence_caught():
    a = [[1, 2, 3, 4], [5, 6]]
    b = [[1, 2, 9, 4], [5, 6]]
    divs = compute_divergences(a, b)
    assert divs == [2, -1]
