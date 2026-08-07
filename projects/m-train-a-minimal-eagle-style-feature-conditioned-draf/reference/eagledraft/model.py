import numpy as np


class TokenOnlyDraftHead:

    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed = rng.randn(vocab_size, embed_dim) * 0.02
        self.head = rng.randn(embed_dim, vocab_size) * 0.02

    def predict_logits(self, token_ids: np.ndarray) -> np.ndarray:
        emb = self.embed[token_ids]
        return np.dot(emb, self.head)


class EagleFeatureDraftHead:

    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, seed: int = 42
    ):
        rng = np.random.RandomState(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embed = rng.randn(vocab_size, embed_dim) * 0.02
        self.proj_feat = rng.randn(hidden_dim, embed_dim) * 0.02
        self.fc = rng.randn(embed_dim * 2, embed_dim) * 0.02
        self.head = rng.randn(embed_dim, vocab_size) * 0.02

    def forward(
        self, token_ids: np.ndarray, hidden_states: np.ndarray
    ) -> np.ndarray:
        t_emb = self.embed[token_ids]
        h_proj = np.dot(hidden_states, self.proj_feat)
        fused = np.concatenate([t_emb, h_proj], axis=-1)
        hidden = np.tanh(np.dot(fused, self.fc))
        return np.dot(hidden, self.head)
