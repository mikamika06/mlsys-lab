import pytest
from chunking.correctness import verify_chunked_prefill_logits


def test_chunked_prefill_correctness_passes():
    full = [[0.1, 0.2, 0.7], [0.4, 0.4, 0.2]]
    chunked = [[0.1, 0.2, 0.7], [0.4, 0.4, 0.2]]
    assert verify_chunked_prefill_logits(full, chunked) is True


def test_chunked_prefill_correctness_mismatched():
    full = [[0.1, 0.2, 0.7]]
    chunked = [[0.9, 0.0, 0.1]]
    assert verify_chunked_prefill_logits(full, chunked) is False
