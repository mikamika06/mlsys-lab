import numpy as np


class MockTokenizer:
    def encode(self, text):
        return [10, 20, 30]


class MockModel:
    def forward(self, tokens):
        np.random.seed(len(tokens))
        return [float(x) for x in np.random.randn(5)]

    def forward_cached(self, tokens):
        return [float(x) for x in np.random.randn(5)]


def get_mock_setup():
    return MockModel(), MockTokenizer(), "hello world"
