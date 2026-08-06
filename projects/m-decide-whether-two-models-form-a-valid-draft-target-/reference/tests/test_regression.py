"""Learner regression tests."""

from specdec.pair import is_valid_draft_target_pair


def test_draft_target_validation():
    target = {
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tokens": ["<pad>", "<s>", "</s>", "a", "b"],
        "add_eos_token": True,
    }
    valid_draft = {
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tokens": ["<pad>", "<s>", "</s>", "a", "b"],
        "add_eos_token": True,
    }
    invalid_draft = {
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tokens": ["<pad>", "<s>", "</s>", "x", "y"],
        "add_eos_token": True,
    }

    assert is_valid_draft_target_pair(valid_draft, target) is True
    assert is_valid_draft_target_pair(invalid_draft, target) is False
