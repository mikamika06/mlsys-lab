import fnmatch


RECIPES = [
    {
        "version": "1.0",
        "stages": [
            {
                "group_1": {
                    "quantization_modifier": {
                        "bits": 4,
                        "group_size": 128,
                        "ignore": ["lm_head"],
                        "target_layers": ["Reversible", "Linear"]
                    }
                }
            }
        ]
    },
    {
        "version": "1.0",
        "stages": [
            {
                "pruning_stage": {
                    "sparsity_modifier": {
                        "target_sparsity": 0.5,
                        "start": 0.0,
                        "end": 0.5
                    }
                },
                "quant_stage": {
                    "quantization_modifier": {
                        "bits": 8,
                        "group_size": 64
                    }
                }
            }
        ]
    },
    {
        "version": "1.0",
        "stages": [
            {
                "stage_a": {
                    "constant_folding_modifier": {}
                },
                "stage_b": {
                    "quantization_modifier": {
                        "bits": 4
                    }
                }
            }
        ]
    },
    {
        "version": "1.0",
        "stages": [
            {
                "mixed_stage": {
                    "sparsity_modifier": {"target_sparsity": 0.3},
                    "quantization_modifier": {"bits": 4}
                }
            }
        ]
    },
    {
        "version": "1.0",
        "stages": [
            {
                "s1": {
                    "quantization_modifier": {"bits": 4}
                }
            },
            {
                "s2": {
                    "smooth_quant_modifier": {"smoothing_strength": 0.5}
                }
            }
        ]
    }
]


def reconstruct_recipe(recipe_dict):
    lines = [f"# Recipe version {recipe_dict.get('version', '1.0')}"]
    lines.append("from llmcompressor.modifiers import *")
    lines.append("recipe = [")
    for stage in recipe_dict.get("stages", []):
        for stage_name, mods in stage.items():
            lines.append(f"    # Stage: {stage_name}")
            for mod_name, mod_args in mods.items():
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in mod_args.items())
                lines.append(f"    {mod_name}({args_str}),")
    lines.append("]")
    return "\n".join(lines)


def validate_ordering(recipe_dict):
    order_map = {
        "constant_folding_modifier": 0,
        "sparsity_modifier": 1,
        "smooth_quant_modifier": 2,
        "quantization_modifier": 3,
    }
    for stage in recipe_dict.get("stages", []):
        for stage_name, mods in stage.items():
            last_idx = -1
            for mod_name in mods.keys():
                idx = order_map.get(mod_name, 99)
                if idx < last_idx:
                    return False
                last_idx = idx
    return True


def count_modules(recipe_dict, module_list):
    matched = set()
    for stage in recipe_dict.get("stages", []):
        for stage_name, mods in stage.items():
            for mod_name, mod_args in mods.items():
                targets = mod_args.get("target_layers", ["*"])
                ignore = set(mod_args.get("ignore", []))
                for mod in module_list:
                    if mod in ignore:
                        continue
                    if any(fnmatch.fnmatch(mod, t) for t in targets):
                        matched.add(mod)
    return len(matched)


MODEL_MODULES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.1.self_attn.q_proj",
    "lm_head"
]
