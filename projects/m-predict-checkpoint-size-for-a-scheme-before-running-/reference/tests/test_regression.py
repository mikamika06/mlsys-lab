import sys
sys.path.insert(0, ".")
from compress.predictor import estimate_checkpoint_size
from compress.chooser import get_supported_schemes
from compress.pipeline import validate_compressed_checkpoint


def test_predictor_sanity():
    config = {"total_weight_params": 1000000}
    scheme = {"bits": 4, "group_size": 128, "meta_overhead_factor": 1.0}
    val = estimate_checkpoint_size(config, scheme)
    assert val > 0


def test_chooser_not_empty():
    for arch in ["ampere", "hopper", "blackwell"]:
        assert len(get_supported_schemes(arch)) > 0


def test_pipeline_validation():
    pass
