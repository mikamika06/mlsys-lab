MERGES_TESTS = [
    ["hello world", "foo bar", "ab cd"],
    ["byte byte", "a b", "test case"],
]

VOCAB_TESTS = [
    ({"has_merges": True, "has_scores": False}, "BPE"),
    ({"has_merges": False, "has_scores": True}, "SentencePiece"),
]

TOKEN_TESTS = [
    ([{"id": 0, "token_type": 1}, {"id": 1, "token_type": 1}, {"id": 2, "token_type": 5}], 2),
    ([{"id": 10, "token_type": 3}, {"id": 11, "token_type": 3}, {"id": 12, "token_type": 2}], 12),
]
