import json
import os


class Tools(list):
    def __str__(self):
        return json.dumps(list(self), separators=(",", ":"), ensure_ascii=False)


class Args(dict):
    def __str__(self):
        return json.dumps(dict(self), separators=(",", ":"), ensure_ascii=False)


def wrap(ctx):
    c = json.loads(json.dumps(ctx))
    if "tools" in c:
        c["tools"] = Tools(c["tools"])
    for m in c.get("messages", []):
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if "arguments" in fn:
                fn["arguments"] = Args(fn["arguments"])
    return c


def load_inputs(fixtures):
    with open(os.path.join(fixtures, "inputs.json"), encoding="utf-8") as f:
        return {k: wrap(v) for k, v in json.load(f).items()}


def run_suite(render, funcs, fixtures):
    """Replay every recorded rendering. Returns (passed, failed)."""
    data = load_inputs(fixtures)
    ollama = os.path.join(os.path.dirname(fixtures), "ollama")
    passed = failed = 0
    for name, base in (("semantics_renderings.json",
                        os.path.join(fixtures, "semantics")),
                       ("renderings.json", ollama)):
        with open(os.path.join(fixtures, name), encoding="utf-8") as f:
            cases = json.load(f)["cases"]
        for case in cases:
            if case.get("error"):
                continue
            with open(os.path.join(base, case["template"]), encoding="utf-8") as f:
                src = f.read()
            try:
                got = render(src, data[case["input"]], funcs)
            except Exception:
                failed += 1
                continue
            if got == case["expected"]:
                passed += 1
            else:
                failed += 1
    return passed, failed
