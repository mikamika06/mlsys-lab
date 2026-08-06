from packeff.waste import compute_waste_percentage
from packeff.crossover import compute_attention_costs
from packeff.batches import compute_batch_counts


def test_waste_bounds():
    lengths = [100, 200, 300]
    max_len = 500
    waste = compute_waste_percentage(lengths, max_len)
    assert 0.0 <= waste <= 100.0


def test_crossover_relation():
    lengths = [128, 256]
    max_len = 512
    costs = compute_attention_costs(lengths, max_len)
    assert costs["packing_cost"] <= costs["padding_cost"]


def test_batch_counts_relation():
    lengths = [100, 150, 200]
    max_len = 512
    counts = compute_batch_counts(lengths, max_len)
    assert counts["packing_batches"] <= counts["padding_batches"]
