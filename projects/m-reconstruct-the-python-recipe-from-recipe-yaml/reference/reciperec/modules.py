import fnmatch


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
