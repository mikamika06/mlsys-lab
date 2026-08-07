import numpy as np


def generate_test_case(seed=42):
    rng = np.random.default_rng(seed)
    shape = [4, 4]
    w0 = rng.standard_normal((2, 4)).astype(np.float32)
    w1 = rng.standard_normal((2, 4)).astype(np.float32)

    metadata = {
        "storage_data": {
            "model.weight": {
                "shape": shape,
                "offsets": [[0, 0], [2, 0]],
                "lengths": [[2, 4], [2, 4]],
                "file_name": ["shard_0.bin", "shard_1.bin"]
            }
        }
    }
    shards = {
        "shard_0.bin": w0.tobytes(),
        "shard_1.bin": w1.tobytes()
    }
    expected = np.vstack([w0, w1])
    return metadata, shards, expected
