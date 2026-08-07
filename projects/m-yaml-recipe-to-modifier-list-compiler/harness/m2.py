import ref

def check(workdir):
    from compiler.recipe import compile_yaml_recipe
    from compiler.goals import validate_six_goals
    out = {"goals_matched": 0.0, "zero_recipe_handled": 0.0}

    try:
        res = validate_six_goals(ref.GOALS)
        if res is True:
            out["goals_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"goals validation failed: {e}"

    try:
        empty_res = compile_yaml_recipe("empty_recipe: null")
        if empty_res == []:
            out["zero_recipe_handled"] = 1.0
        else:
            out["_note"] = f"empty recipe returned {empty_res} instead of []"
    except Exception as e:
        out["_note"] = f"empty recipe raised {e}"

    return out
