def test_mask_mod_correctness():
    from flexmask.repair import corrected_mask_mod
    assert corrected_mask_mod(10, 5, 32) is True
    assert corrected_mask_mod(5, 10, 32) is False
