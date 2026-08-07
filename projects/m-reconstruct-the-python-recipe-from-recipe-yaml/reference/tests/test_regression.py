import sys

sys.path.insert(0, ".")
from reciperec.ordering import validate_ordering
from reciperec.modules import count_modules
from reciperec.reconstruct import reconstruct_recipe


def test_validate_ordering_rejects_bad_order():
    bad_recipe = {
        "version": "1.0",
        "stages": [{
            "s": {
                "quantization_modifier": {"bits": 4},
                "sparsity_modifier": {"target_sparsity": 0.5}
            }
        }]
    }
    assert validate_ordering(bad_recipe) is False


def test_count_modules_respects_ignore():
    recipe = {
        "version": "1.0",
        "stages": [{
            "s": {
                "quantization_modifier": {
                    "bits": 4,
                    "target_layers": ["*"],
                    "ignore": ["lm_head"]
                }
            }
        }]
    }
    mods = ["layer.0.linear", "lm_head"]
    assert count_modules(recipe, mods) == 1


def test_reconstruct_recipe_output_format():
    recipe = {
        "version": "1.0",
        "stages": [{
            "s1": {
                "quantization_modifier": {"bits": 4}
            }
        }]
    }
    out = reconstruct_recipe(recipe)
    assert "quantization_modifier" in out
    assert "bits=4" in out
