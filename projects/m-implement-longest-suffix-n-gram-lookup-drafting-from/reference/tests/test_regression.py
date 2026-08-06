import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_lookup.simulate import simulate


def test_speculative_decoding_is_lossless():
    prompt = [1, 2, 3]
    target = [1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]
    res = simulate(prompt, target, max_n=2, max_draft_len=3)
    assert res["generated"] == target, "Decoding modified the sequence"


def test_speculative_decoding_never_exceeds_baseline_steps():
    prompt = [1]
    target = [1, 2, 3, 4, 5]
    res = simulate(prompt, target, max_n=1, max_draft_len=2)
    baseline = len(target) - len(prompt)
    assert res["steps"] <= baseline, "Took more steps than autoregressive"
