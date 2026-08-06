from flexmask.cache import MaskCache, DummyBlockMask


def test_mask_cache_shape_invalidation():
    """Verify that MaskCache invalidates or rejects cached entries when sequence lengths change."""
    cache = MaskCache(max_capacity=4)

    def builder_512():
        return DummyBlockMask((512, 512), 128, 10, "causal")

    def builder_1024():
        return DummyBlockMask((1024, 1024), 128, 36, "causal")

    m1, hit1 = cache.get_or_create((512, 512), 128, "causal", builder_512)
    assert not hit1
    assert m1.shape == (512, 512)

    m2, hit2 = cache.get_or_create((1024, 1024), 128, "causal", builder_1024)
    assert not hit2
    assert m2.shape == (1024, 1024)

    m3, hit3 = cache.get_or_create((512, 512), 128, "causal", builder_512)
    assert hit3
    assert m3.shape == (512, 512)
    assert m3 is m1
    assert m3 is not m2
