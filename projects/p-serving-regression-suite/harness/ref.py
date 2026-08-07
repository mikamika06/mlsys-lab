import importlib.util
import os
import sys

def run_tests(workdir):
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        return None

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    spec = importlib.util.spec_from_file_location("test_regression_mod", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return False

    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None

    for fn in fns:
        try:
            fn()
        except AssertionError:
            return False
        except Exception:
            return False

    return True

def survives(workdir):
    res = run_tests(workdir)
    return res is True

def apply_regression(workdir, reg_idx):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import serving.engine as eng
    good_init = eng.Engine.__init__
    good_tokenize = eng.Engine.tokenize
    good_generate = eng.Engine.generate

    def restore():
        eng.Engine.__init__ = good_init
        eng.Engine.tokenize = good_tokenize
        eng.Engine.generate = good_generate

    if reg_idx == 1:
        counter = [0]
        def bad_gen1(self, prompt, max_tokens=10, stop_tokens=None, seed=None):
            import random
            if stop_tokens is None: stop_tokens = []
            counter[0] += 1
            rng = random.Random(counter[0])
            out = []
            for _ in range(max_tokens):
                t = rng.choice(list(self.vocab.values()))
                out.append(t)
                if t in stop_tokens: break
            text = " ".join(self.rev.get(t, "<unk>") for t in out)
            return {"prompt": prompt, "tokens": out, "text": text, "usage": {"prompt_tokens": len(self.tokenize(prompt)), "completion_tokens": len(out)}}
        eng.Engine.generate = bad_gen1

    elif reg_idx == 2:
        def bad_tok(self, text):
            return [self.vocab.get(w, 0) for w in text.split()]
        eng.Engine.tokenize = bad_tok

    elif reg_idx == 3:
        def bad_gen3(self, prompt, max_tokens=10, stop_tokens=None, seed=None):
            import random
            if stop_tokens is None: stop_tokens = []
            eff_seed = seed if seed is not None else self.seed
            rng = random.Random(eff_seed + len(prompt))
            out = []
            for _ in range(max_tokens):
                t = rng.choice(list(self.vocab.values()))
                out.append(t)
            text = " ".join(self.rev.get(t, "<unk>") for t in out)
            return {"prompt": prompt, "tokens": out, "text": text, "usage": {"prompt_tokens": len(self.tokenize(prompt)), "completion_tokens": len(out)}}
        eng.Engine.generate = bad_gen3

    elif reg_idx == 4:
        def bad_gen4(self, prompt, max_tokens=10, stop_tokens=None, seed=None):
            import random
            if stop_tokens is None: stop_tokens = []
            eff_seed = seed if seed is not None else self.seed
            rng = random.Random(eff_seed + len(prompt))
            out = []
            for _ in range(max_tokens + 1):
                t = rng.choice(list(self.vocab.values()))
                out.append(t)
                if t in stop_tokens: break
            text = " ".join(self.rev.get(t, "<unk>") for t in out)
            return {"prompt": prompt, "tokens": out, "text": text, "usage": {"prompt_tokens": len(self.tokenize(prompt)), "completion_tokens": len(out)}}
        eng.Engine.generate = bad_gen4

    elif reg_idx == 5:
        def bad_gen5(self, prompt, max_tokens=10, stop_tokens=None, seed=None):
            import random
            if stop_tokens is None: stop_tokens = []
            eff_seed = seed if seed is not None else self.seed
            rng = random.Random(eff_seed + len(prompt))
            out = []
            for _ in range(max_tokens):
                t = rng.choice(list(self.vocab.values()))
                out.append(t)
                if t in stop_tokens: break
            text = " ".join(self.rev.get(t, "<unk>") for t in out)
            return {"prompt": prompt, "tokens": out, "text": text}
        eng.Engine.generate = bad_gen5

    return restore
