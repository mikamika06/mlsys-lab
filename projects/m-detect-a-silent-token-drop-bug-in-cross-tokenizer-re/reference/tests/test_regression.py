import sys
sys.path.insert(0, ".")
from realign.aligner import align_tokens

def test_no_silent_drop():
    draft = [101, 102, 103, 104]
    vocab = {101: 1001, 102: 1002, 103: 1003, 104: 1004}
    res = align_tokens(draft, vocab)
    assert len(res) == len(draft), f"dropped tokens: got {len(res)}, expected {len(draft)}"

def test_token_count_conservation():
    draft = [1, 2, 3, 4, 5]
    vocab = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
    res = align_tokens(draft, vocab)
    assert len(res) == len(draft)
