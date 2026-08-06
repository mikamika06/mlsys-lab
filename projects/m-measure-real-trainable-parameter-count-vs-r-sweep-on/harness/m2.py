import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    from lorameasure.expansion import expand_target_modules
    from lorameasure.params import count_trainable_params

    out = {"expansion_matched": 0.0}
    ok = 0
    total = len(ref.MODELS)

    for idx, model in enumerate(ref.MODELS):
        want_expanded = ref.expand_target_modules(model, "all-linear")
        got_expanded = expand_target_modules(model, "all-linear")

        if sorted(got_expanded) == want_expanded:
            want_params = ref.count_trainable_params(model, want_expanded, r=8)
            got_params = count_trainable_params(model, got_expanded, r=8)
            if want_params == got_params:
                ok += 1

    if ok == total:
        out["expansion_matched"] = 1.0
    else:
        out["_note"] = f"Matched {ok}/{total} model expansion configs"

    return out
