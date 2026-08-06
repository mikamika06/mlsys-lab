import numpy as np
import pytest
from embedrun.safety import validate_embedding_pipeline


def test_valid_normalized_embeddings():
    rng = np.random.default_rng(42)
    embs = rng.normal(size=(10, 16))
    embs /= np.linalg.norm(embs, axis=-1, keepdims=True)
    assert validate_embedding_pipeline(embs, embs) is True


def test_invalid_unnormalized_embeddings():
    rng = np.random.default_rng(42)
    embs = rng.normal(size=(10, 16))
    with pytest.raises(ValueError):
        validate_embedding_pipeline(embs, embs)
