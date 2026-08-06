from jaccl.perf import min_microbatches_for_bubble


def test_min_microbatches():
    mb_2_ranks = min_microbatches_for_bubble(2, 0.20)
    mb_4_ranks = min_microbatches_for_bubble(4, 0.20)
    assert mb_2_ranks == 4
    assert mb_4_ranks == 12
    assert mb_4_ranks > mb_2_ranks
