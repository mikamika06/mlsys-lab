import os
import importlib.util

def check(workdir):
    def _run(path):
        spec = importlib.util.spec_from_file_location("test_regression", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
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

    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_starvation": 0.0,
        "catches_budget": 0.0,
        "faults_caught": 0.0
    }

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        return out

    import packer.packer as pmod
    good_step = pmod.Packer.step

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    # Fault 1: Starve active decodes if prefills consume the entire budget
    def starved_step(self):
        ans = {}
        b = self.token_budget
        new_waiting = []
        for r in self.waiting_prefill:
            if b > 0:
                take = min(b, r.prefill_left)
                ans[r.rid] = take
                r.prefill_left -= take
                b -= take
                if r.prefill_left == 0:
                    r.is_decode = True
                    self.active_decodes.append(r)
                else:
                    new_waiting.append(r)
            else:
                new_waiting.append(r)
        self.waiting_prefill = new_waiting

        new_decodes = []
        for r in self.active_decodes:
            if r.is_decode and b >= 1 and r.rid not in ans:
                ans[r.rid] = 1
                b -= 1
            new_decodes.append(r)
        self.active_decodes = new_decodes
        return ans

    pmod.Packer.step = starved_step
    try:
        out["catches_starvation"] = 0.0 if _survives(path) else 1.0
    finally:
        pmod.Packer.step = good_step

    # Fault 2: Exceed token budget completely
    def greedy_step(self):
        ans = {}
        for r in self.active_decodes:
            ans[r.rid] = 1
        for r in self.waiting_prefill:
            ans[r.rid] = r.prefill_left
            r.prefill_left = 0
            r.is_decode = True
            self.active_decodes.append(r)
        self.waiting_prefill = []
        return ans

    pmod.Packer.step = greedy_step
    try:
        out["catches_budget"] = 0.0 if _survives(path) else 1.0
    finally:
        pmod.Packer.step = good_step

    out["faults_caught"] = out["catches_starvation"] + out["catches_budget"]
    return out
