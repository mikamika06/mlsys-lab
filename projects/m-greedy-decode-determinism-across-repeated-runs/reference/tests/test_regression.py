import sys
sys.path.insert(0, ".")
from decoder.pipeline import run_greedy_decode, measure_latencies
from decoder.metrics import compute_match_fraction, analyze_latency_ratio


class MockTokenizer:
    def encode(self, text):
        return [1, 2, 3]


class MockModel:
    def forward(self, tokens):
        return [0.1 * i for i in range(10)]

    def forward_cached(self, tokens):
        return [0.1 * i for i in range(10)]


def test_greedy_determinism():
    model = MockModel()
    tokenizer = MockTokenizer()
    seqs = run_greedy_decode(model, tokenizer, "test", runs=3)
    assert compute_match_fraction(seqs) == 1.0


def test_latency_difference():
    cold = [1000, 1050, 1020]
    reused = [500, 510, 505]
    ratio = analyze_latency_ratio(cold, reused)
    assert ratio >= 1.2
