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
