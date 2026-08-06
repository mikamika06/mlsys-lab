import ref
import numpy as np

def check(workdir):
    from speculative.metrics import expected_accepted_length_tree
    from speculative.tree import verify_tree_sample

    out = {"tree_rel_err": 1.0, "path_match": 0.0}
    errors = []
    path_matches = 0

    for case in ref.TEST_TREE_CASES:
        parents = case["parents"]
        probs = case["probs"]
        want_exp = ref.expected_accepted_length_tree(parents, probs)
        try:
            got_exp = expected_accepted_length_tree(parents, probs)
            err = abs(got_exp - want_exp) / max(1e-6, abs(want_exp))
            errors.append(err)
        except Exception as e:
            out["_note"] = f"Error computing tree expected length: {type(e).__name__}: {e}"
            return out

        rng = np.random.default_rng(123)
        accepts = (rng.uniform(0, 1, size=len(parents)) < np.array(probs)).tolist()
        want_path = ref.verify_tree_sample(parents, accepts)
        try:
            got_path = verify_tree_sample(parents, accepts)
            if list(got_path) == list(want_path):
                path_matches += 1
        except Exception as e:
            out["_note"] = f"Error running verify_tree_sample: {type(e).__name__}: {e}"
            return out

    out["tree_rel_err"] = float(np.mean(errors))
    out["path_match"] = float(path_matches) / len(ref.TEST_TREE_CASES)
    return out
