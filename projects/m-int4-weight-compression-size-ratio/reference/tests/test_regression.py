import sys

sys.path.insert(0, ".")
from compression.ratio import size_ratio
from compression.footprint import memory_footprint
from compression.awq import perplexity_delta


def test_size_ratio_accounts_for_metadata():
    r_with_meta = size_ratio(1048576, 4, 128, 16)
    r_without_meta = 16.0 / 4.0
    assert r_with_meta < r_without_meta, "Size ratio failed to account for metadata overhead"


def test_memory_footprint_scaling():
    fp_small = memory_footprint(1048576, 4, 128, 16)
    fp_large = memory_footprint(2097152, 4, 128, 16)
    assert fp_large > fp_small, "Memory footprint did not scale with weight count"


def test_perplexity_delta_positive_improvement():
    delta = perplexity_delta(15.5, 9.1)
    assert delta > 0, "Perplexity delta should be positive when AWQ improves over data-free"
