import sys
sys.path.insert(0, ".")
from onnxconfig.variants import make_variants
from onnxconfig.validator import validate

def test_variants_count():
    specs = [{"name": "test_model", "max_batch": 16, "input_shape": [1, 10], "output_shape": [1, 10]}]
    for spec in specs:
        variants = make_variants(spec)
        assert len(variants) == 3

def test_variants_validity():
    specs = [{"name": "test_model", "max_batch": 16, "input_shape": [1, 10], "output_shape": [1, 10]}]
    for spec in specs:
        for v in make_variants(spec):
            assert validate(v) is True

def test_max_batch_positive():
    specs = [{"name": "test_model", "max_batch": 16, "input_shape": [1, 10], "output_shape": [1, 10]}]
    for spec in specs:
        for v in make_variants(spec):
            assert v["max_batch_size"] > 0
