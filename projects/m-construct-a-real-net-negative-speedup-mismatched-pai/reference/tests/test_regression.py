import spec_fail.pathology as p

def test_longest_match_preferred():
    seq = [1, 2, 3, 4, 2, 5, 1, 2]
    drafted = p.prompt_lookup_draft(seq, 2)
    assert drafted == [3, 4], f"Expected [3, 4], got {drafted}"
