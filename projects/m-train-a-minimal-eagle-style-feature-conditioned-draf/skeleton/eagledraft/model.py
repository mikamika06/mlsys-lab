import numpy as np


class TokenOnlyDraftHead:

    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
        raise NotImplementedError

    def predict_logits(self, token_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class EagleFeatureDraftHead:

    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, seed: int = 42
    ):
        raise NotImplementedError

    def forward(
        self, token_ids: np.ndarray, hidden_states: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError
