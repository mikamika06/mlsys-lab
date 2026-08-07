import sys
sys.path.insert(0, ".")
from compiler.recipe import compile_yaml_recipe

def test_compiled_recipe_schema():
    yaml_snippet = "quant_stage:\n  quant_modifiers:\n    - target_names: [q_proj]\n      bits: 4\n      group_size: 128"
    mods = compile_yaml_recipe(yaml_snippet)
    for m in mods:
        if "bits" in m:
            assert isinstance(m["bits"], int), f"bits field must be an integer, got {type(m['bits'])}"

def test_zero_recipe_returns_list():
    res = compile_yaml_recipe("empty_recipe: null")
    assert isinstance(res, list), "zero recipe must compile to a list"
    assert len(res) == 0, "zero recipe must contain no modifiers"
