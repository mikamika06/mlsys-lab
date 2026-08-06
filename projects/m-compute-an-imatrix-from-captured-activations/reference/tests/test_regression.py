import numpy as np
from imatrix.merge import merge_imatrices


def test_merge_weighting():
    """Verify that imatrix merging uses sample count weights correctly."""
    shard1 = {
        "count": 100,
        "data": {"layer1": np.array([1.0, 2.0], dtype=np.float32)}
    }
    shard2 = {
        "count": 300,
        "data": {"layer1": np.array([5.0, 6.0], dtype=np.float32)}
    }
    result = merge_imatrices([shard1, shard2])
    expected = np.array([4.0, 5.0], dtype=np.float32)
    assert result["count"] == 400
    assert np.allclose(result["data"]["layer1"], expected, atol=1e-5)
