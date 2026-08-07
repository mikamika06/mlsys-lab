def test_savings_measured_in_tokens():
    from prefix_cache.cache import PrefixCache
    c = PrefixCache(16)
    c.insert("t1", [(1,), (2,)], [10, 20])
    matched = c.match("t1", [(1,), (2,)])
    assert len(matched) == 2
    assert c.saved_tokens == 32, "Savings must be calculated in tokens, not blocks"
