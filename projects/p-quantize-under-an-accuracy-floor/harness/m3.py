import ref
import sys


def check(workdir):
    m = {"recipe_compared": 0.0}
    sys_path_orig = list(sys.path)
    try:
        sys.path.insert(0, workdir)
        import quant.recipe as q_recipe

        w = ref.ToyModel().layers[0]
        q8 = q_recipe.quantize_uniform(w, 8)
        q4 = q_recipe.quantize_uniform(w, 4)
        if q8 and q4 and "weights" in q8 and "weights" in q4:
            m["recipe_compared"] = 1.0
    except Exception:
        pass
    finally:
        sys.path[:] = sys_path_orig
    return m
