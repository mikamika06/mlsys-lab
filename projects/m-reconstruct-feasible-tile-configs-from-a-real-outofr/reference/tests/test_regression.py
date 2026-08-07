import sys

sys.path.insert(0, ".")
from tile_recon.predict import predict_out_of_resources
from tile_recon.reconstruct import reconstruct_configs


def test_prediction_monotonicity_with_block_size():
    low = predict_out_of_resources(49152, 4, 32, 32, 2, 1024)
    high = predict_out_of_resources(49152, 4, 128, 128, 4, 1024)
    assert not low, "small config should fit"
    assert high, "large config should exceed shared memory"


def test_reconstruction_filters_correctly():
    candidates = [
        {"BLOCK_M": 16, "BLOCK_N": 16, "stages": 2, "overhead": 100},
        {"BLOCK_M": 128, "BLOCK_N": 128, "stages": 4, "overhead": 100}
    ]
    valid = reconstruct_configs(49152, 4, candidates)
    assert len(valid) == 1
    assert valid[0]["BLOCK_M"] == 16
