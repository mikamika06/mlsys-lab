import importlib.util
import os

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_token_bug": 0.0}
    if not os.path.isfile(path):
        return out

    import prefix_cache.cache as cache

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_match = cache.PrefixCache.match

    def buggy_match(self, tenant_id, blocks):
        hashes = self.compute_hashes(tenant_id, blocks)
        self.clock += 1
        curr = self.root
        matched = []
        for h in hashes:
            if h in curr.children:
                curr = curr.children[h]
                curr.last_access = self.clock
                matched.append(curr.block_id)
            else:
                break
        self.saved_tokens += len(matched)
        return matched

    cache.PrefixCache.match = buggy_match
    try:
        out["catches_token_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        cache.PrefixCache.match = good_match

    return out
